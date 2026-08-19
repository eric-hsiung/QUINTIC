import itertools
import fractions
import numpy as np
import z3
import random
import copy
import time
from collections import deque
from datastructures import BijectiveIndexMapping, TableList, EquivalenceClass
from stacks import Checkpoint, AtomicAssumptionStack, LocalAssumptionStack, GlobalAssumptionStack
from lstar_utils import concat, define_domain, get_vars, range_prefixes, convert_to_equivalence_classes, assert_rows_equal
from lstar_utils import ValuationModel, SymbolicHypothesis

class SymbolicObservationTable:
    """
    This observation table is based on symbolic entries in the table:

    The symbolic observation table is the tuple <S,E,T;C,G>, and it
    consists of the following:
        * S is the set of prefixes
        * E is the set of suffixes
        * T is the sequence-to-output function
        * C is the set of constraints
        * G is the context (set of variables)
    """
    ## [TESTED][DONE]
    def __init__(self, sigma_I, sigma_O, preference_query, equivalence_query,
            prefix = None,
            suffix = None,
            constraints = {
                "eq": set(), "ineq": set(), "EC": dict(), "repr": dict(),
                "seq_eq": set(), "seq_ineq": set(), "seq_EC": dict(), "seq_repr": dict(),
                "seq_sym": set(),
            },
            context = dict(),
            valuation=None,
            gamma_val=None,
            use_closed_and_consistent_obj=True,
            use_variable_equivalence_obj=True,
            use_ids=True):
        """
        Here, we define the semantics of the different constraint sets:

        We let
            "eq"
            "ineq"
            "EC"
            "repr"

        refer to equality, inequality, equivalence classes, and reprensentatives with respect to variables.
        These are DERIVED from the seq_* constraints.

        We let
            "seq_eq"
            "seq_ineq"
            "seq_EC"
            "seq_repr"
            "seq_sym"

        refer to equality, inequality, equivalence classes, and representatives with respect to values, as represented
        symbolically by sequences.


        Prefix set S and suffix set E are each represented by pairs of dictionaries,
        which map sequences to indices, and indices to sequences.
            - .prefix_set: (\Sigma)^* -> row idx
            - .inv_prefix_set: row idx -> (\Sigma)^*
            - .suffix_set: (\Sigma)^* -> col idx
            - .inv_suffix_set: col idx -> (\Sigma)^*

        Context: Maps: Sequence -> Variable
        """
        self.verbose = True
        self.use_ids = use_ids
        self.use_closed_and_consistent_obj = use_closed_and_consistent_obj
        self.use_variable_equivalence_obj = use_variable_equivalence_obj
        ## prefixes, suffxies, and prefix-alpha sets
        self.prefix_set = BijectiveIndexMapping()
        self.suffix_set = BijectiveIndexMapping()
        self.prefix_alpha_set = BijectiveIndexMapping()

        ## We represent the upper and lower parts of the table separately, as lists of lists
        self.table_upper = TableList()
        self.table_lower = TableList()
        ## Store a reference to the alphabet
        self.sigma_I = sigma_I
        self.sigma_O = sigma_O
        self.preference_query = preference_query
        self.equivalence_query = equivalence_query

        ## Constraints and Context:
        self.constraints = constraints
        self.context = context
        self.gamma_var = z3.Real("g")
        self.gamma_val = gamma_val

        def __summation__(s):
            return z3.Sum(tuple(self.context[x] for x in range_prefixes(s)))

        def __discounted_summation__(s):
            return z3.Sum(tuple(self.context[x]*(self.gamma_var**(len(x))) for x in range_prefixes(s)))

        def __product__(s):
            return z3.Product(tuple(self.context[x] for x in range_prefixes(s)))

        def __classification__(s):
            return self.context[s]

        ## Internal sequence decomposition function
        def decompose(a1, b1):
            a = a1[:len(a1)-1]
            x = a1[len(a1)-1:]
            b = b1[:len(b1)-1]
            y = b1[len(b1)-1:]
            return a, x, b, y

        ## Internal lookup function
        def prefQ(m, p, q):
            ## TODO: Handle looking up representative
            return (m[p] > m[q]) - (m[p] < m[q])

        def __DT_summation__(m, A, B):
            ## We record the total ordering in seq2idx. We assume that seq2idx includes only Value Representatives.
            ## Given a pair sequences p = ax and q = by, with len(x) = 1, len(y) = 1, len(a) >= 0, and len(b) >= 0,
            ## we perform a test that determines whether:
            ## (a) T(p) > T(q), (b) T(p) < T(q), (c) T(p) == T(q), or (d) inconclusive
            ## based on a decision tree involving:
            ## 1. prefQ(a, b), prefQ(ax, by)
            ## 2. prefQ(by,b), prefQ(ax, a)
            a, x, b, y = decompose(A, B)
            ## Level 1:
            X, Y = prefQ(m, A, B), prefQ(m, a, b)
            if abs(X+Y) == 2:
                ## Go to level 2:
                X, Y = prefQ(m, A, a), prefQ(m, B, b)
                if abs(X+Y) == 2:
                    return None
                else:
                    return (X > Y) - (X < Y)
            else:
                return (X > Y) - (X < Y)
       
        def __DT_discounted_summation__(m, A, B):
            ## We record the total ordering in seq2idx. We assume that seq2idx includes only Value Representatives.
            ## Given a pair sequences p = ax and q = by, with len(x) = 1, len(y) = 1, len(a) >= 0, and len(b) >= 0,
            ## we perform a test that determines whether:
            ## (a) T(p) > T(q), (b) T(p) < T(q), (c) T(p) == T(q), or (d) inconclusive
            ## based on a decision tree involving:
            ## 1. prefQ(a, b), prefQ(ax, by)
            ## 2. prefQ(by,b), prefQ(ax, a)
            a, x, b, y = decompose(A, B)

            if len(a) == len(b):
                return __DT_summation__(m, A, B)
            else: ## only Level 2 is valid when |a| != |b|
                X, Y = prefQ(m, A, a), prefQ(m, B, b)
                if abs(X+Y) == 2:
                    return None
                else:
                    return (X > Y) - (X < Y)
        
        def __DT_product__(m, A, B):
            ## We record the total ordering in seq2idx. We assume that seq2idx includes only Value Representatives.
            ## Given a pair sequences p = ax and q = by, with len(x) = 1, len(y) = 1, len(a) >= 0, and len(b) >= 0,
            ## we perform a test that determines whether:
            ## (a) T(p) > T(q), (b) T(p) < T(q), (c) T(p) == T(q), or (d) inconclusive
            ## based on a decision tree involving:
            ## 1. prefQ(a, b), prefQ(ax, by)
            ## 2. prefQ(by,b), prefQ(ax, a)
            return __DT_summation__(m, A, B)

        def __DT_classification__(m, A, B):
            return prefQ(m, A, B)
 
        ## Valuation Model
        self.models = {
            "sum": __summation__,
            "discountsum": __discounted_summation__,
            "prod": __product__,
            "classification": __classification__,
        }

        ## Corresponding Decision Tree Model
        self.DT_models = {
            "sum": __DT_summation__,
            "discountsum": __DT_discounted_summation__,
            "prod": __DT_product__,
            "classification": __DT_classification__,
        }

        if valuation not in self.models:
            raise ValueError("valuation must be a key in self.models")

        self.valuation_model = self.models[valuation]
        self.query_total_order = self.DT_models[valuation]

        self.SymbolicEvaluationModel = ValuationModel(valuation, gamma_var=self.gamma_var, gamma_val=self.gamma_val)
        self.symbolic_eval = self.SymbolicEvaluationModel.symbolic_eval
        self.generate_cex_constraint = self.SymbolicEvaluationModel.generate_cex_constraint

        self.G = z3.Int("G") ## Variable for tracking dimension (i.e. number of variables)

        ## Unique Sequence Tracking
        self.old_entries = set()
        self.old_sorted = deque()

        ## Unknown Relations
        self.unknown_relations = list()

        ## Checkpoint Stack and Assumption Stack
        self.checkpoint_stack = deque()
        self.model_stack = deque()
        self.global_stack = GlobalAssumptionStack()
        self.past_hypotheses = dict()
        ## We will be utilizing MaxSAT
        self.solver = z3.Optimize()
        ## Ensure Core Minimization
        #self.solver.set("sat.core.minimize", True)
        self.assertions_object = z3.Solver()

        self.initialize_table(prefix, suffix)
        self.budget_sf = 1
        self.max_budget_sf = 1

        self.__exp__num_pref_queries = 0
        self.__exp__num_equi_queries = 0
        self.__exp__total_max_sat_obj_time = 0
        self.__exp__count_max_sat_obj_solves = 0
        self.__exp__cex_lengths = []

        self.__exp__events = []
        self.__exp__event_id = -1

    def print(self, s):
        if self.verbose:
            print(s)

    def record_event(self, event_type):
        self.__exp__event_id += 1
        event_id = self.__exp__event_id
        ## Number of States
        num_states = self.table_upper.num_unique_rows()
        ## Number of Classes
        num_classes = 0
        if len(self.model_stack) > 0:
            EC = self.last_model()
            reps = set()
            for k, v in EC.items():
                reps.add(v.repr())
            num_classes = len(reps)

        ## Number of Variables in the Table
        num_variables = len(self.context)
        ## Table Size
        ru, cu = self.table_upper.shape()
        rl, cl = self.table_lower.shape()
        
        self.__exp__events.append((event_id, event_type, num_states, num_classes, num_variables, ru, cu, rl, cl))

    def record_cex_length(self, cex):
        self.__exp__cex_lengths.append(len(cex))

    def experimental_data(self):
        """
        Returns total number of preference queries that were made
        Total number of unique sequences that were tested
        Total number of equivalence classes
        Number of unique variables in observation table
        Dimensions of upper table
        Dimensions of lower table
        Number of equivalence queries made
        Tuple of the CEX lengths in order that they were received
        Event list
        """
        EC = self.constraints["EC"]
        reps = set()
        for k, v in EC.items():
            reps.add(v.repr())

        return (
            self.__exp__num_pref_queries,
            len(self.constraints["ineq"]),
            len(self.context), 
            len(reps),
            self.__exp__total_max_sat_obj_time,
            self.__exp__count_max_sat_obj_solves,
            len(self.unknown_relations),
            len(self.table_upper.get_entry_set() | self.table_lower.get_entry_set()),
            self.table_upper.shape(),
            self.table_lower.shape(),
            self.__exp__num_equi_queries,
            tuple(self.__exp__cex_lengths),
            self.__exp__events,
        )

    ## [TESTED][DONE]
    def initialize_table(self, prefix, suffix):
        """
        Preconditions: Both tables should be size (0,0).
        Postconditions: After this initialization process,
            1 <= len(prefix set) <= 1 + len(prefix)
            1 <= len(suffix set) <= 1 + len(suffix)
            len(prefix_alpha set) == len(sigma^I)*len(prefix set)
            table_upper.shape() == (len(prefix set) , len(suffix set))
            table_lower.shape() == (len(prefix_alpha set) , len(suffix set))
        """
        ## NOTE: Individual prefixes and individual suffixes are sequences.
        ## Therefore, we require that each element of "prefix" and "suffix" must be a tuple.
        ## Initialize the prefix set
        empty_seq = tuple()
        if prefix is None:
            self.prefix_set.add(empty_seq)
        else:
            self.prefix_set.add(empty_seq)
            for el in prefix:
                if el is None:
                    self.prefix_set.add(empty_seq)
                elif isinstance(el, (tuple,list,str)):
                    self.prefix_set.add(tuple(el))
                else:
                    self.prefix_set.add((el,))

        ## Initialize the suffix set
        if suffix is None:
            self.suffix_set.add(empty_seq)
        else:
            self.suffix_set.add(empty_seq)
            for el in suffix:
                if el is None:
                    self.suffix_set.add(empty_seq)
                elif isinstance(el, (tuple,list,str)):
                    self.suffix_set.add(tuple(el))
                else:
                    self.suffix_set.add((el,))

        ## Initialize the prefix_alpha set based on the intialized prefixes
        for el in self.prefix_set.forward:
            for alpha in self.sigma_I:
                self.prefix_alpha_set.add(concat(el, alpha))

        ## Adjust the table column sizes
        suffix_sz = len(self.suffix_set)
        self.table_upper.ncols = suffix_sz
        self.table_lower.ncols = suffix_sz

        ## Initialize the table entries to None
        for _ in range(len(self.prefix_set)):
            self.table_upper.append_row()
        for _ in range(len(self.prefix_alpha_set)):
            self.table_lower.append_row()

        ## Check Constraints and Context:
        self.print(f"Initial Constraints:\n")
        self.print(self.constraints)
        self.print(f"Initial Context:\n")
        self.print(self.context)

    ## DONE: Add self.constraints["repr"] as a dict() of repr -> value
    ## Need to reflect this in the:
    ## [DONE] Unification function
    ## [DONE] Equivalence Query procedure
    def make_hypothesis(self):
        self.print(" >> Table is Unified, Closed, and Consistent <<\n")
        self.print(">>>===  Constructing Symbolic Hypothesis ===<<<\n")
        self.print(self)

        ## NOTE: In REMAP with Sums, we split make_hypothesis into 3 portions:
        ## 1. Creating the symbolic hypothesis (states and transitions and initial state)
        ## 2. A constraint check on the symbolic hypothesis
        ## 3. If this passes, then construction of the concrete hypothesis.
        ## Construct the states
        states = dict()
        init_state = None
        for prefix, ridx in self.prefix_set.forward.items():
            row = tuple(self.table_upper.get_row(ridx))
            if row not in states:
                states[row] = prefix

            if ridx == 0:
                init_state = row

        ## Construct the transition function
        delta = dict()
        for row, prefix in states.items():
            delta[row] = dict()
            for letter in self.sigma_I:
                ridx = self.prefix_alpha_set.get(concat(prefix, letter))
                delta[row][letter] = tuple(self.table_lower.get_row(ridx))

        ## NOTE: We now need to ensure the symbolic hypothesis does not violate any constraints.
        ## The constraints come from:
        ## 1. PrefQ constraints which have been unified according to known variable equivalence classes and assumption stack
        ## 2. Symbolic CEX constraints which induce concrete CEX constraints based on the current hypothesis being used
       
        ## Make a record of proposing this hypothesis 
        current_hypo = SymbolicHypothesis(init_state, states, delta, self.sigma_I)
        if len(current_hypo) not in self.past_hypotheses:
            self.past_hypotheses[len(current_hypo)] = [current_hypo]
        else:
            self.past_hypotheses[len(current_hypo)].append(current_hypo)

        ## Pass constraints to the SMT solver so that we can get a hypothesis output function:
        ## NOTE: Due to unification, all of the constraints will be in terms of representatives.
        ## Representatives ultimately are derived from known equivalence classes of variables and the assumption stack.
        self.print(" >>>=== Solving for Concrete Values ===<<<\n")
        ## Take the existing constraints C_k
       
        ## NOTE: Push 
        self.solver.push()
        self.solver.add(self.assertions_object.assertions())
        ## Add assumed equivalence classes
        self.add_equivalence_classes_to_solver(negate=False) ## NOTE: Adds the cached equivalence classes currently in self.constraints["EC"]
        ## Add the CEX tests
        self.add_cex_to_solver(
            (states, self.sigma_I, self.sigma_O, init_state, delta, None)
        )
        sat_unsat = self.solver.check()

        if sat_unsat == z3.sat:
            self.print("Found a satisfying solution for concrete hypothesis")
            
            model = self.solver.model()
            ## NOTE: pop
            self.solver.pop()

            ## Construct the output function
            output_fnc = dict()
            for row, prefix in states.items():
                ridx = self.prefix_set.get(prefix)
                var_entry = self.table_upper.get_entry(ridx, 0)
                ## Convert from internal Z3 representation to Python
                output_fnc[row] = model[var_entry].as_long()

            self.__exp__num_equi_queries += 1
            return z3.sat, (states, self.sigma_I, self.sigma_O, init_state, delta, output_fnc)
        else: ## Handle UNSAT
            ## NOTE: If we have unsat, then we need to backtrack on the assumption stack.
            self.print("No satisfying solution for concrete hypothesis")
            ## NOTE: pop
            self.solver.pop()
            ## This means the equivalence classes are not valid. These will be negated in the symbolic_fill(backtrack=True)
            
            return z3.unsat, None

    def new_concrete_hypothesis(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        self.print(" >>>=== Resolving for Concrete Values ===<<<\n")
        ## NOTE: We need to negate the current output_fnc
        self.solver.add(z3.Not(z3.And(list(vector[0] == value for vector, value in output_fnc.items()))))

        ## NOTE: Push 
        self.solver.push()
        self.solver.add(self.assertions_object.assertions())
        ## Add assumed equivalence classes
        self.add_equivalence_classes_to_solver(negate=False) ## NOTE: Adds the cached equivalence classes currently in self.constraints["EC"]
        ## Add the CEX tests
        self.add_cex_to_solver(
            (states, self.sigma_I, self.sigma_O, init_state, delta, None)
        )
        self.print(self.solver)
        sat_unsat = self.solver.check()
        if sat_unsat == z3.sat:
            self.print("Found a satisfying solution for concrete hypothesis")
            
            model = self.solver.model()
            ## NOTE: pop
            self.solver.pop()

            ## Construct the output function
            output_fnc = dict()
            for row, prefix in states.items():
                ridx = self.prefix_set.get(prefix)
                var_entry = self.table_upper.get_entry(ridx, 0)
                ## Convert from internal Z3 representation to Python
                output_fnc[row] = model[var_entry].as_long()

            self.__exp__num_equi_queries += 1
            return z3.sat, (states, self.sigma_I, self.sigma_O, init_state, delta, output_fnc)
        else: ## Handle UNSAT
            ## NOTE: If we have unsat, then we need to backtrack on the assumption stack.
            self.print("No satisfying solution for concrete hypothesis")
            ## NOTE: pop
            self.solver.pop()
            ## This means the equivalence classes are not valid. These will be negated in the symbolic_fill(backtrack=True)
            
            return z3.unsat, None

    def __fmt_equivalence_classes__(self):
        EC = self.constraints["EC"]
        reps = set()
        for k, v in EC.items():
            reps.add(v.repr())

        s = f" --> NUMBER OF EQUIVALENCE CLASSES: {len(reps)}\n"
        counter = 1
        for v in reps:
            s += f"EC{counter}:\n{EC[v]}\n"
            counter+=1

        return s

    def __fmt_repr_values__(self):
        RP = self.constraints["repr"]
        s = f" --> NUMBER OF KNOWN REPR VALUES: {len(RP)}\n"
        s += f"  REPR VALUES:\n"
        for k, v in RP.items():
            s+= f"    {k}=={v}\n"
        return s

    def __fmt_ineqs__(self):
        INEQ = self.constraints["ineq"]
        s = f" --> NUMBER OF INEQUALITIES: {len(INEQ)}\n"
        s += f"  INEQUALITIES:\n"
        for v in INEQ:
            s += f"    {v}\n"
        return s

    def __str__(self):
        """
        Print out everything that we can about the observation table so that we can debug this issue
        """
        s = f"PREFIXES:\n{self.prefix_set}\nPREFIX_ALPHA_SET:\n{self.prefix_alpha_set}\nSUFFIXES:\n{self.suffix_set}\n"
        ## Also double check number of unique variables in the table
        table_entry_set = self.table_upper.get_entry_set() | self.table_lower.get_entry_set()
        s+= f"TABLE ENTRIES:\n"
        s+= f"  --> NUMBER OF UNIQUE ENTRIES: {len(table_entry_set)}\n"
        s+= f"  TABLE ENTRY SET: {table_entry_set}\n"
        s+= f"TABLE_UPPER:\n{self.table_upper}\nTABLE_LOWER:\n{self.table_lower}\n"
        s+= f"EQUIVALENCE_CLASSES: {self.__fmt_equivalence_classes__()}\n"
        #s+= f"KNOWN REPRESENTATIVE VALUES: {self.__fmt_repr_values__()}\n"
        #s+= f"UNIFIED CONSTRAINTS: {self.__fmt_ineqs__()}\n"
        return s
        

    ## DONE
    ## [TESTED][DONE]
    def is_closed(self):
        """
        The Observation Table is closed iff rowset(table_lower) is a subset of
        rowset(table_upper)

        Returns: a tuple --
            Bool, prefix_alpha

            If Bool is True, prefix_alpha is None
            If Bool is False, prefix_alpha is a tuple
        """
        ## Does the upper table contain all the rows of the lower table?
        status, ridx = self.table_upper.contains_rows(self.table_lower)
        prefix_alpha = None
        ## If not, then find the row index of a row in the lower table that is not
        ## in the upper table, and return it, so that in the next step, we can add that
        ## row to the upper table.
        if not status:
            prefix_alpha = self.prefix_alpha_set.get_idx(ridx)
        return status, prefix_alpha

    ## DONE
    ## [TESTED][DONE]
    def is_consistent(self):
        """
        The purpose of the consistency check: determine whether the transitions
        are deterministic. Remember, the unique rows of S determine the states
        Q.
        Let q \\in Q. If q, a -> q_1, and q, a -> q_2, for q_1 != q_2, then this
        must mean that even though currently q = row(s_1) = row(s_2), row(s_1)
        and row(s_2) must actually be different.
        """
        ## For every pair of identical rows in table_upper:
        row_to_idx = dict()
        nonunique_rows = set()
        for ridx in range(self.table_upper.nrows):
            trow = tuple(self.table_upper.get_row(ridx))
            if trow not in row_to_idx:
                row_to_idx[trow] = [ridx]
            else:
                row_to_idx[trow].append(ridx)
                nonunique_rows.add(trow)

        ## All the rows are unique
        if len(nonunique_rows) == 0:
            return True, None
        ## The rows are not unique -- check all pairs for each row
        for trow in nonunique_rows:
            ## Get pairs
            for s1_idx, s2_idx in itertools.combinations(row_to_idx[trow], 2):
                s1 = self.prefix_set.get_idx(s1_idx)
                s2 = self.prefix_set.get_idx(s2_idx)

                for a in self.sigma_I:
                    s1a = concat(s1, a)
                    s2a = concat(s2, a)
                    r1idx = self.prefix_alpha_set.get(s1a)
                    r2idx = self.prefix_alpha_set.get(s2a)
                    ## Are the two rows symbolically equivalent?
                    status, result = self.table_lower.check_row_equivalence(r1idx, r2idx)
                    if not status:
                        suffix = self.suffix_set.get_idx(result)
                        return status, concat(tuple((a,)), suffix)
        return True, None

    ## DONE
    ## [TESTED][DONE]
    def expand_prefixes(self, prefix_alpha, expand_table=True):
        """
        This is the part of the algorith where we have already identified that
        row(s1*a) is not found in rowspace(S).

        This adds the prefix_alpha in prefix_alpha to the prefix set,
        and is used in the Closed test:

        add the string s.a to S, then extend T to (S u S.A).E
        This means, |S| has increased by 1, and |S.A| has increased by (|A| - 1)

        NOTE: prefix_alpha is a sequence; and we need it to be hashable
        """
        if expand_table:
            ## Add the row from the lower table to the upper table
            ridx_lower = self.prefix_alpha_set.get(prefix_alpha)
            new_row = self.table_lower.get_row(ridx_lower)
            self.table_upper.append_row(new_row)

        ## Add to the prefix set
        self.prefix_set.add(prefix_alpha)
        
        ## Expand the prefix_alpha set
        for alpha in self.sigma_I:
            self.prefix_alpha_set.add(concat(prefix_alpha, alpha))

        ## Return here; a symbolic fill will be called afterwards

    ## DONE
    ## [TESTED][DONE][TESTED in test_is_consistent]
    def expand_suffixes(self, alpha_suffix):
        """
        This expands the alpha_suffix (adds it to the suffix set). This occurs
        in the consistency check

        NOTE: We modify this function to also ensure that all prefixes of alpha_suffix
        are also added to the table. This is because we need to ensure that
        (S*E) itself is prefix-closed
        """
        ## Add a*e to E, and expand the number of columns in the table.
        for seq in range_prefixes(alpha_suffix):
            self.suffix_set.add(seq)
        ## Return here; a symbolic fill will be called afterwards

    def forced_expansion(self, counterexamples):
        """
        This forcefully expands the table.
        """
        ## One method: add counterexamples, if any.
        ## If the counterexamples have been exhausted, then choose to inspect the upper and lower tables.
        
        options = self.prefix_alpha_set.difference(self.prefix_set)
        ## Try to add the shortest option
        length = None
        shortest = None
        for opt in options:
            if length is None:
                length = len(opt)
                shortest = opt
            else:
                if len(opt) < length:
                    length = len(opt)
                    shortest = opt

        self.expand_prefixes(shortest, expand_table=False)
        
    def generate_minimize_states_objective(self):
        ## Minmize the number of states in the table
        ur, uc = self.table_upper.shape()
        lr, lc = self.table_lower.shape()

        ## NOTE: There are {(U(U-1) + L(L-1)) // 2 + UL} terms in the objective
       
        count_equalities = (z3.If(0 == 0, 0, 0) + 
            sum(z3.If(assert_rows_equal(self.table_upper.get_row(r1), self.table_upper.get_row(r2)), 1, 0) for r1, r2 in itertools.combinations(range(ur), 2)) +
            sum(z3.If(assert_rows_equal(self.table_lower.get_row(r1), self.table_lower.get_row(r2)), 1, 0) for r1, r2 in itertools.combinations(range(lr), 2)) +
            sum(z3.If(assert_rows_equal(self.table_upper.get_row(r1), self.table_lower.get_row(r2)), 1, 0) for r1, r2 in itertools.product(range(ur), range(lr))))
        return count_equalities, ((ur*(ur-1) + lr*(lr-1))//2 + ur*lr)

    def generate_prefer_consistent_table_objective(self):
        """
        For a table to be consistent, generate the requirement that if a pair of rows in UPPER are equivalent, then each of
        those rows must transition identitically.
        """

        ## 1. Generate all possible pairs of rows in upper which can be equal.
        ur, uc = self.table_upper.shape()
        lr, lc = self.table_lower.shape()

        consistency_terms = list()
        for u1_idx, u2_idx in itertools.combinations(range(ur), 2):
            ## If eq_pair (a pair of rows in upper are equivalent)
            u1 = self.table_upper.get_row(u1_idx)
            u2 = self.table_upper.get_row(u2_idx)
            U = assert_rows_equal(u1, u2)
            ## then all their transitions must be equivalent
            ## 2. Look up the sequence index for the pair:
            s1 = self.prefix_set.get_idx(u1_idx)
            s2 = self.prefix_set.get_idx(u2_idx)
            equiv_transitions = list()
            for sigma in self.sigma_I:
                t1 = concat(s1, sigma)
                t2 = concat(s2, sigma)
                l1_idx = self.prefix_alpha_set.get(t1)
                l2_idx = self.prefix_alpha_set.get(t2)
                l1 = self.table_lower.get_row(l1_idx)
                l2 = self.table_lower.get_row(l2_idx)
                ## The equivalent transitions should result in the same row (row equivalence)
                L = assert_rows_equal(l1, l2)
                equiv_transitions.append(L)
            L = z3.And(equiv_transitions)
            consistency_terms.append(z3.Implies(U, L))
        ## 3. Require each implication to be true
        consistent = z3.And(consistency_terms)
        return consistent

    def generate_prefer_closed_table_objective(self):
        """
        For a table to closed, generate the requirement that for each for in LOWER, it must equal to any of the rows in UPPER.
        r == r1 or r == r2 or r == r3 ... etc
        """
        ur, uc = self.table_upper.shape()
        lr, lc = self.table_lower.shape()
        
        ## Require that rows of the lower table must be members of the rows of the upper table
        closed = z3.And(list(z3.Or(list(assert_rows_equal(self.table_lower.get_row(lo_idx), self.table_upper.get_row(hi_idx)) for hi_idx in range(ur))) for lo_idx in range(lr)))
        return closed

    def exclude_all_previous_hypotheses(self):
        ## Generates symbolic constraints which would exclude the current table being isomorphic any of the previous hypotheses
        if len(self.past_hypotheses) > 0:
            nrows, ncols = self.table_upper.shape()
            ## Only exclude prior hypotheses that have |Q|<=nrows
            neg_iso_constraints = list()
            for k, hypothesis_list in self.past_hypotheses.items():
                if k <= nrows:
                    for H in hypothesis_list:
                        neg_iso_constraints.append(z3.Not(self.generate_isomorphism_constraints(H)))
            self.solver.add(z3.And(neg_iso_constraints))

    def generate_isomorphism_constraints(self, sym_hyp):
        """ 
        Generates the constraints K that would make the current table equivalent to sym_hyp, and then returns K.
        sym_hyp is a SymbolicHypothesis object
        We assume the number of rows in the upper table is greater than or equal to the number of states in sym_hyp
        """

        ## NOTE: Remember that in an finite automaton, each state represents a set of prefix sequences, which if processed
        ## starting from the initial state, would eventually all transition into the same state.
        ## Therefore, we gather sets of prefixes. Any prefix in an equivalence class of prefixes must then have its indexed
        ## row be equivalent to every other prefix in the same equivalence class.
        empty = self.prefix_set.get_idx(0)
        prefix_classes = {sym_hyp.q0: [empty]}
        checked = set()
        checked.add(empty)

        N = len(self.prefix_alpha_set)

        ## NOTE: prefix_alpha_set is prefix closed, so by processing it, also includes all elements in prefix_set
        for s_idx in range(N):
            idx = N - 1 - s_idx
            s = self.prefix_alpha_set.get_idx(idx)

            if s not in checked:
                q = sym_hyp.q0

                for c_idx in range(len(s)):
                    letter = s[c_idx][0]
                    prefix = s[0:c_idx+1]
                    q = sym_hyp.delta[q][letter]
                    
                    if prefix not in checked:
                        if q not in prefix_classes:
                            prefix_classes[q] = [prefix]
                        else:
                            prefix_classes[q].append(prefix)
                        checked.add(prefix)

        ## Now iterate through the prefix classes and determine which rows must be equivalent to one another.
        row_constraints = list()
        for q, prefix_list in prefix_classes.items():
            ## We can only create a constraint if the prefix list has at least 2 elements.
            if len(prefix_list) > 1:
                ## Just make all the rows equivalent to the first row
                ## Check if this prefix is in both upper and lower tables
                rep = prefix_list[0] ## Use first prefix as the representative
                if q == sym_hyp.q0:
                    rep = prefix_list[1]
                idx = self.prefix_alpha_set.get(rep)
                row_rep = self.table_lower.get_row(idx)
                ## First, make all constraints for all the rows in the lower table.
                for prefix in prefix_list:
                    if prefix in self.prefix_alpha_set and prefix != rep:
                        idx = self.prefix_alpha_set.get(prefix)
                        row = self.table_lower.get_row(idx)
                        row_constraints.append(assert_rows_equal(row_rep, row))
                    ## Just require all the upper rows with the prefix to also be equal to the rep
                    ## There must exist an upper rep in the list. Find it.
                    if prefix in self.prefix_set:
                        idx = self.prefix_set.get(prefix)
                        row = self.table_upper.get_row(idx)
                        row_constraints.append(assert_rows_equal(row_rep, row))
        return z3.And(row_constraints)

    ## DONE
    ## [TESTED][DONE]
    def symbolic_fill(self, backtrack=False, init_constraint=None):
        is_real_backtrack = backtrack
        if self.use_ids:
            print(f"Current Budget: SNAPSHOT: {self.budget_sf} / {self.max_budget_sf}")
            if self.budget_sf == 0:
                ## If no more budget, force a backtrack
                backtrack = True
                print(f"Current Budget: EXHAUSTED, FORCING BACKTRACK {self.budget_sf} / {self.max_budget_sf}")

        if not backtrack:
            self.print(" ===> Symbolic Fill <===")
            ## Make sure to resize the tables
            self.table_upper.resize(len(self.prefix_set), len(self.suffix_set))
            self.table_lower.resize(len(self.prefix_alpha_set), len(self.suffix_set))

            ## TODO: Somehow cache the old entries
            newentries = set()
            oldentries = set()

            ## Create new fresh variables when required
            for suffix, cidx in self.suffix_set.forward.items():
                for prefix, ridx in self.prefix_set.forward.items():
                    ps = concat(prefix, suffix) ## Look up by sequence, since identical sequences should have same output
                    if ps not in self.context:
                        sz = len(self.context)
                        self.context[ps] = z3.Int(f"v{sz}")
                        newentries.add(ps)
                    #else:
                    #    ## The entry is either already an old entry, or it is already in the new entry list
                    #    oldentries.add(ps)
                    self.table_upper.set_entry(ridx, cidx, self.context[ps])
                for prefix, ridx in self.prefix_alpha_set.forward.items():
                    ps = concat(prefix, suffix)
                    if ps not in self.context:
                        sz = len(self.context)
                        self.context[ps] = z3.Int(f"v{sz}")
                        newentries.add(ps)
                    #else:
                    #    oldentries.add(ps)
                    self.table_lower.set_entry(ridx, cidx, self.context[ps])

            num_pref_queries = 0

            ## Compare the new entries
            sorted_new_entries, new_prefs = self.quicksort(list(newentries))
            num_pref_queries += new_prefs

            ## NOTE: That here, len(sorted_new_entries) <= len(newentries) because we include
            ## only one instance of sequences that are equivalent to one another

            ## Compare the new and old entries
            ## Merge the lists
            merged = deque()
            while len(sorted_new_entries) > 0 and len(self.old_sorted) > 0:
                p1 = sorted_new_entries[-1]
                p2 = self.old_sorted[-1]
                p = self.preference_query(p1, p2)
                self.update_constraint_set_value(p, p1, p2) ## Send the pair to the contraint set
                num_pref_queries += 1

                if p > 0:
                    s = sorted_new_entries.pop()
                    merged.appendleft(s)
                elif p < 0:
                    s = self.old_sorted.pop()
                    merged.appendleft(s)
                else:
                    ## They are considered equal, so we prefer taking from old_sorted
                    s = self.old_sorted.pop()
                    _ = sorted_new_entries.pop()
                    merged.appendleft(s)

            while len(sorted_new_entries) > 0:
                s = sorted_new_entries.pop()
                merged.appendleft(s)

            while len(self.old_sorted) > 0:
                s = self.old_sorted.pop()
                merged.appendleft(s)

            self.old_sorted = merged

            ## NOTE: Previously, we used to explicitly make n(n-1)/2 preference queries to the teacher.
            ## Pair combinations -- pairs of (new, new) entries, and pairs of (new, old) entries
            #for p1, p2 in itertools.chain(itertools.combinations(newentries, 2), itertools.product(newentries, oldentries)):
            #    ## Query the teacher
            #    p = self.preference_query(p1, p2) ## Send the seqquences to the preference query
            #    ## Update the constraint set
            #    self.update_constraint_set_variable(p, p1, p2) ## Send the pair to the contraint set
            #    num_pref_queries += 1
            
            self.__exp__num_pref_queries += num_pref_queries
            print(f"Executed {num_pref_queries} preference queries to the teacher")
            
            ## NOTE: Now, self.old_sorted contains unique SEQUENCES sorted by their values, where the
            ## value is given by a function V(T, sequence). From this total ordering over sequence values,
            ## we can extract all n(n-1)/2 comparsons of the values of sequences. Let us assume that we have
            ## all those constraints stored in C. Let that constraint information be the set of known constraints.
            ##
            ## From this set of constraints, we need to be able to infer information about whether any arbitrary
            ## pair of constraints on variables is permitted.
            ##
            ## NOTE: Below, we need to track both ECs over sequence values, and ECs over sequence variables.

            ## TODO: Step 1. Compute ALL known constraints based on the total ordering from the sequences.
            ## Let all these constraints be represented by K.
            ## NOTE: Not sure if this step is strictly necessary.
            ## The total ordering is implicit and can be recovered from transitive property chaining on the comparisons
            ## obtained from the sorting algorithm.

            ## TODO: Step 2. If possible, identify whether a pair of the variables in the table are equal or not equal.
            ## There are 2 ways to do this:
            ##
            ## (A) Given K, add(K) to solver, push(), then add(vi == vj), then check(), then pop(), and then
            ##     do a push(), add(vi!=vj), check(), then pop(). If both checks() are SAT, then we don't know, and will
            ##     have to make a hypothesis about it. Otherwise, if exactly one of them is SAT and the other is UNSAT,
            ##     then we know with certainty the relation between vi and vj. If both are UNSAT, then we know there is
            ##     a conflict somewhere in the constraints.
            ##     
            ##     Cost:
            ##     Using this method requires n(n-1)/2 * 2 checks() of the solver, or n(n-1) solver checks() (from scratch).
            ##
            ## (B) Given K, use a lookup table to determine if we can deduce whether two variables are equal, one is greater
            ##     than the other, or one is less than the other. Specifically, we can attempt to deduce whether
            ##     (a) T(so) > T(s'o'), (b) T(so) < T(s'o'), (c) T(so) == T(s'o'), (d) inconclusive
            ##     by looking at the coordinates of (prefQ(so,s'o'), preQ(s,s')) and (prefQ(so, s), prefQ(s'o', s')) one at a time.
            ##
            ##     Cost:
            ##     Therefore, for each variable pair, we perform at most 4 lookups. There are n(n-1)/2 pairs, so we perform a total
            ##     of at most 2n(n-1) lookups.
            ##
            ## Regardless of whether (A) or (B) is used, we store the known equality / inequality relations in K', and we store
            ## the Unknown relations in U'. Actually, we should perform a substitution step here using equivalence classes. Then
            ## everything in K', K, and U' will be in terms of the representatives.
            ##
            ## Below, we utilize approach (B)
            
            seq2idx = self.build_seq2idx()
            ## NOTE: Every sequence should exist in this dictionary
            ## Sequence equivalence classes are updated here if they are missing.

            for p1, p2 in itertools.chain(itertools.combinations(newentries, 2), itertools.product(newentries, self.old_entries)):
                p = self.query_total_order(seq2idx, p1, p2)
                if p is not None:
                    ## Constraint set specifically for variable relations
                    self.update_constraint_set_variable(p, p1, p2)
                else:
                    ## Unknown relations for a pair of variables. Assume they are equivalent, for the next process.
                    self.unknown_relations.append((True, self.context[p1], self.context[p2]))
            
            ## Update the old entries now, AFTER we have selected all the combinations.
            self.old_entries.update(newentries)
          
            ## [DONE]: Remove this reset to make the constraints cumulative 
            ## self.solver.reset()
            ## Add all variables in the context to the solver
            ## Make sure we explicitly add the domain of the variables, as we weren't doing this before.
            ## [DONE]: Just add the domain whenever the variables are created
            for ps in newentries:
                self.solver.add(define_domain(self.context[ps], self.sigma_O))

            ## NOTE: If the table from the previous iteration included equality assumptions, then 
            ## the following will include those possibly incorrectly unified variables.
            ## Therefore, we need to ensure that we keep a non-unified table around.
            ## In particular, it is always safe to perform unification over known equivalent variables,
            ## but we need to ensure that we 
            #self.solver.add(self.constraints["seq_eq"])
            #self.solver.add(self.constraints["seq_ineq"])
        else: ## Do the backtrack
            self.print(" ===> Symbolic Fill Backtrack <===")
            mk = self.pop_model() ## Pop the old model
            ## Add the negated equivalence class instead of the specific negated solution mk
            if is_real_backtrack:
                self.add_equivalence_classes_to_solver(negate=True, EC=mk) ## NOTE: Adds negated equivalence classes currently in self.constraints["EC"]
            else:
                self.add_equivalence_classes_to_solver(negate=True, EC=mk, solver=self.assertions_object)
            
            ## [DONE]: We need to also revert to the (k-2)th \hat{O} table.
            self.pop_checkpoint()
            self.record_event("backtrack")
            if self.use_ids:
                self.budget_sf += 1
                print(f"Current Budget: BACKTRACKED: {self.budget_sf} / {self.max_budget_sf}")

        if init_constraint is not None:
            self.print(f"Initial Constraint is {init_constraint}")
            self.solver.add(z3.Or(self.context[init_constraint[0]] == init_constraint[1]))
            self.solver.add(z3.Or(self.G > 0)) ## Context should contain at least one variable.
            if self.gamma_val is not None:
                self.solver.add(z3.Or(self.gamma_var == self.gamma_val))

        self.print(self)
        SAT = False
        m = None ## Model to be pushed (this represents a solution to the constraints).
        dict_of_values = None
        while not SAT:
            self.print(" ===> MaxSAT <===")
            ## NOTE: Add MaxSAT condition -- create as many equalities between variables as possible
            ## [DONE]: Use z3.Optimize() instead of z3.Solver()
            ## [DONE]: REPLACE the optimization objective whenever new constraints are obtained.
            ## NOTE: push() and pop() also correctly push and pop objectives
            self.solver.push()
            self.solver.add(self.assertions_object.assertions())
            ## 1. Exclude options which would immediately generate a previous hypothesis
            self.exclude_all_previous_hypotheses()
            ## Try to add counterexamples explicitly
            self.add_cex_to_solver(None)
            ## 2. If possible, prefer a solution which immediately results in a closed table.
            if self.use_closed_and_consistent_obj:
                obj_closed = self.generate_prefer_closed_table_objective()
                obj_consistent = self.generate_prefer_consistent_table_objective()
                ## Ideally, we want a table that is both closed and consistent. We rank:
                ## closed and consistent > closed (not consistent) > not closed
                obj_minimal_states, base = self.generate_minimize_states_objective()

                ## Since closed, consistent, and minimal states are potentially conflicting, we purposely create a priority within
                ## the objective. To minimize the number of states independently, we attempt to maximize the number of row equalities
                ## in the table. There are base = (U + L choose 2) such equalities. We then rank solution by closed and consistency.
                ## In particular, closed and consistent solutions are always preferred over closed, but not consistent solutions, and
                ## soltuions which are not closed and not consistent are preferred the least.

                self.solver.maximize(z3.If(obj_closed, z3.If(obj_consistent, 2*(base+1), 1*(base+1)), 0) + obj_minimal_states)
            ## Currently uses lexographic solving order -- try to maximize row equivalences first, then do the remaining unknowns.
            ## 3. If a closed table is possible, then try to minimize the number of states.
            ## Similarly, if a closed table is not possible, try to minimize the number of states.
            #self.solver.maximize(count_equalities)
            ## 4. Finally, try to maximize the number of variable equalities.
            if self.use_variable_equivalence_obj:
                count_equalities = z3.If(0 == 0, 0, 0) + sum( z3.If(v1 == v2, 1, 0) for _, v1, v2 in self.unknown_relations)
                self.solver.maximize(count_equalities)
            ## [DONE]: Also need to enforce the PREVIOUS solution \Lambda_{k-1}; the new solution needs
            ## to remain consistent with the previous solution.
            ## NOTE: Each concrete solution corresponds to an equivalence class, so just add the previously assumed
            ## equivalence class
            self.solver.add(z3.Or(self.G == len(self.context)))
            if len(self.model_stack) > 0:
                mk = self.last_model()
                self.add_equivalence_classes_to_solver(negate=False, EC=mk)
                #self.solver.add( z3.And( list( k.__call__() == mk[k] for k in mk ) ) )
            self.print(self.solver)

            __t__ = time.perf_counter_ns()
            sat_unsat = self.solver.check()
            self.__exp__total_max_sat_obj_time += time.perf_counter_ns() - __t__
            self.__exp__count_max_sat_obj_solves += 1
            ## If SAT, then we can get self.solver.model() -- this corresponds with the kth solution

            fails_representative_check = False
            if sat_unsat == z3.sat:
                ## If SAT, then we need to check the total number of representatives.
                m = self.solver.model()

                self.print("MaxSAT Solution exists for constraints")
                self.print(m)

                ## Construct the equivalence classes from this solution. At the same time, count the
                ## number of classes.
                dict_of_values = dict() ## Concrete Value -> EquivalenceClass
                ## [DONE]: This should only be for variables that are CURRENTLY IN THE TABLE. It is possible for
                ## the constraints to be in terms of variables that are NOT in the current table.
                for variable in self.table_upper.get_entry_set() | self.table_lower.get_entry_set():
                    val = m[variable]
                    if val not in dict_of_values:
                        dict_of_values[val] = EquivalenceClass({variable})
                    else:
                        dict_of_values[val].add(variable)

                ## NOTE: Because this is MaxSAT, then this means that if the number of representatives
                ## has exceeded |Sigma^O|, then any other solutions that exist will ALSO exceed the
                ## cardinality of Sigma^O.
                fails_representative_check = bool(len(dict_of_values) > len(self.sigma_O))
                self.print(f"Number of representatives: {len(dict_of_values)}")

            self.solver.pop() ## NOTE: Pops the optimization objective. We do this so that the negated solutions are not popped.

            if sat_unsat != z3.sat or fails_representative_check: ## Both cases mean no satisfying solutions exist
                ## If UNSAT, then we need to backtrack. The (k-1)th solution is invalid.
                ## NOTE: We do not need to revert the constraints, since preference queries should be self-consistent
                self.print("UNSAT or MaxSAT Solution failed representative check")
                if self.use_ids and self.budget_sf == self.max_budget_sf:
                    self.max_budget_sf += 1
                    self.budget_sf = self.max_budget_sf
                    ## TODO: Reset equivalence class constraints.
                    self.assertions_object.reset()
                    print(f"Current Budget: BUDGET_INCREASE {self.budget_sf} / {self.max_budget_sf}")

                if len(self.model_stack) == 0:
                    ## If the constraints are UNSAT, and there was no previously enforced equivalence class set,
                    ## then we need to expand the table.
                    ## NOTE: Force a table expansion immediately.
                    return True

                mk = self.pop_model()
                
                ## [DONE]: Add explicitly dis-allowed solutions from previously disproven models
                ## [DONE]: We need to retrieve the (k-1)th solution, and append it as NOT (k-1)th solution
                if mk is not None:
                    #expr = z3.Not( z3.And( list( k.__call__() == mk[k] for k in mk ) ) )
                    self.add_equivalence_classes_to_solver(negate=True, EC=mk)
                    #self.print(expr)
                    #self.solver.add(expr)
                
                ## [DONE]: We need to also revert to the (k-2)th \hat{O} table.
                self.pop_checkpoint()
                ## Go back to the solver check.
                SAT = False
                self.record_event("backtrack")
                if self.use_ids:
                    self.budget_sf += 1
                    print(f"Current Budget: BACKTRACKED: {self.budget_sf} / {self.max_budget_sf}")
            else:
                ## NOTE: Found a satisfactory solution to the constraints
                self.print("MaxSAT Solution passes check")
                SAT = True

        ## Convert dict_to_values to equivalence classes of variables
        equivalence_classes = dict()
        for value, ec in dict_of_values.items():
            for v in ec.members:
                equivalence_classes[v] = ec

        ## Push the current solution
        self.push_model(equivalence_classes)
        ## Push the current state of the non-unified table
        self.push_checkpoint()
        if self.use_ids:
            self.budget_sf -= 1
            print(f"Current Budget: BUDGET_SPENT {self.budget_sf} / {self.max_budget_sf}")

        ## TODO: Step 3. Simulate a conceptual hypothesis CNF h, whose clauses are based on the elements in U'. Select one at random,
        ## then propagate any implications according to equivalence and non-equivalence classes. Then, check if any conflicts arise
        ## via SAT checking: both K and K' should already be added to the solver. We just need to push(h) and check for conflicts:
        ##
        ## Assumption Stack in Z3:
        ##   K
        ##   K'
        ##   push()
        ##   h
        ## check()
        ## If unsat, then while last assumption is !=, pop assumption !=. Pop the = assumption, then push the != assumption.
        ##
        ## else if SAT, then just do another push() h' from U'.
        ## If U' is empty, then continue on with unification.
        ##
        ## Optional -- First perform partial unification using variable equivalence classes?

        ## Perform unification
        self.unification(equivalence_classes)
        ## return (table and model have already been placed on the stack)
        return False

    def symbolic_fill_backtrack(self, local_assumption_stack, atomic_assumption_stack, unknown_relations):
        ## NOTE: If we EVER get to this stage, our only option is to backtrack to the previous symbolic fill.
        ## To do this, we need to:
        ## Step 1. Empty the current atomic assumption stack, and empty the solver. At this point, this
        ##         just means perform the last pop().
        atomic_assumption_stack.pop()
        self.solver.pop()
        ## Step 2. Clear the current unknown_relations.
        unknown_relations.clear()
        ## Step 3. Find the most recent **non-empty** atomic assumption stack in the local assumption stack that has AT LEAST 1 equality...
        ##         because if it is all inequalities, then we just need to backtrack on it again.
        while len(local_assumption_stack) > 0:
            if local_assumption_stack.last().is_empty():
                local_assumption_stack.pop()
            else:
                if local_assumption_stack.last().all_false():
                    local_assumption_stack.pop()
                else:
                    atomic_assumption_stack = local_assumption_stack.pop()
                    break
        
        ## Step 4. Perform backtracking on that atomic assumption stack to set up an appropriate unknown_relations stack.
        while len(atomic_assumption_stack) > 0 and (not atomic_assumption_stack.last()[0]):
            b, vv1, vv2 = atomic_assumption_stack.pop()
            unknown_relations.append((not b, vv1, vv2))

        if len(atomic_assumption_stack) > 0:
            b, vv1, vv2 = atomic_assumption_stack.pop()
            unknown_relations.append((not b, vv1, vv2))
        
        ## Step 5. Restore the table checkpoint.
        atomic_assumption_stack.checkpoint.restore(self)
        
        ## Step 6. Break out of this loop and return the beginning of the while( unknown_relations > 0) loop.
        return atomic_assumption_stack

    def check_representatives(self, unknown_relations, atomic_assumption_stack, local_assumption_stack):
        """
        Returns whether the fails_representative_check passes or fails:

        This check is only valid at a leaf node.

        Returns False if len(unknown_relations) > 0

        Returns True if len(unknown_relations) == 0 AND the number of representatives
        that would exist after unification is greater than len(self.sigma_O)

        Returns False otherwise.
        """

        if len(unknown_relations) > 0:
            return False

        self.print(f"Checking equivalence class effects due to assumptions...")

        ## Step 1. Combine the existing "eqs" and atomic assumptions into a single set.
        equivalences = set()
        for eq in self.constraints["eq"]:
            equivalences.add(eq)
        
        for b, vv1, vv2 in atomic_assumption_stack:
            if b:
                equivalences.add(vv1 == vv2)
        
        for b, vv1, vv2 in local_assumption_stack:
            if b:
                equivalences.add(vv1 == vv2)

        for b, vv1, vv2 in self.global_stack:
            if b:
                equivalences.add(vv1 == vv2)

        ## Step 2. Make a copy of the currently known ECs and RPs from the previous symbolic fill
        EC = copy.deepcopy(self.constraints["EC"]) ## Dictionary of equivalence classes -- Variable -> EquivalenceClass
        RP = copy.deepcopy(self.constraints["repr"]) ## Dictionary of representatives -- Variable -> Value
       
        ## Step 3. Compute the updated equivalence classes 
        convert_to_equivalence_classes(equivalences, EC, RP, print_fnc=self.print)
        #reps = set()
        #for k, v in EC.items():
        #    reps.add(v.repr())
        
        self.print(f"There are {len(RP)} representatives.")

        return len(RP) > len(self.sigma_O)

    def process_unsat_core(self, local_assumption_stack, unsat_core):
        """
        NOTE: We enforce that the UNSAT core be minimal. If M is a minimal unsatisfiable set, then
        this means M becomes satisfiable when any element of M is removed.

        Since the unsat core comes in the form a collection of assumptions made during a symbolic fill,
        this means that the assumptions in M are not compatible. However, we should rarely encounter a
        case where M has a size greater than 1... this is because in a symbolic fill, progress is only
        permitted when the set of assumptions made are all satisfiable. If not satisfiable, then the
        symbolic fill executes backtracking.

        Therefore, one way to process the minimal unsat core would be to identify the assumption in the core
        that appears the earliest....

        From M, pick the one that appears earliest in the list of assumptions that have been made. If we pick
        the latest one, then our algorithm will be complete, because it will help eliminate a subtree.

        NOTE: The approach described above is not quite correct. Consider the case where the minimal UNSAT 
        core contains exactly one unsatisfiable assumption, called _a_. Suppose that our assumption stack of
        assumptions starts with _a_; let the rest of the assumptions in the stack be called _a'_. This means
        that if _a_ was the minimal UNSAT core, then _a'_ is also an UNSAT core (only one of _a_ or _a'_ can
        be satisfied at the same time). However, if we insist on _a_ being false due to the UNSAT core, then
        this implies that we will completely ignore the rest of the remaining search tree where _a_ was true.
        This means that the search will NOT be complete, since there a possibility that the correct answer
        lies in that part of the tree (which we will be forced to skip).

        This means that we should not take the UNSAT core existance as verbatim.

        The existance of the UNSAT core just means that we should just perform standard small step backtracking
        to maintain the completeness property of the search.
        """
        ## NOTE: We are removing the steps for processing the UNSAT core, because 
        ##       the current method for processing the UNSAT core  makes the search INCOMPLETE!
        ## Step 1. Extract the unsat core constraints.
        ## For each assumption in the stack that IS in unsat_sore, add to the conjunction C
        #C = list()
        #for assumption in self.global_stack.assumptions():
        #    if z3.Bool(str(assumption)) in unsat_core:
        #        C.append(assumption)
            
        #for assumption in local_assumption_stack.assumptions():
        #    if z3.Bool(str(assumption)) in unsat_core:
        #        C.append(assumption)
        #
        #assertion = z3.Not(z3.And(C))

        ## NOTE: We iterate through the assumptions in reverse order, and stop once we find one that is
        ## contained in the UNSAT core.
        ## At this point, the local assumption stack has not been added the global assumption stack
        ## so we will check the local assumption stack first.

        atomic_assumption_stack = None
        target_statement = None
        target_assumption = None
        found = False
       
        ## Initialize the setup 
        self.print("STACK_TRACK: Pushed Local Assumption Stack in process_unsat_core")
        self.global_stack.push(local_assumption_stack)

        #while len(self.global_stack) > 0:
        #    local_assumption_stack = self.global_stack.pop()
        #    while len(local_assumption_stack) > 0:
        #        atomic_assumption_stack = local_assumption_stack.pop()
        #        ## Check if this atomic_assumption_stack contains the relation in the unsat core.
        #        ## If so, then we have identified the area of the assumption tree to backtrack in.
        #        for assumption in reversed(atomic_assumption_stack.assumption_stack):
        #            b, v1, v2 = assumption
        #            statement = v1 == v2 if b else v1 != v2
        #            if z3.Bool(str(statement)) in unsat_core:
        #                target_statement = statement
        #                target_assumption = assumption
        #                found = True
        #                break
        #        if found:
        #            break
        #    if found:
        #        break
        self.print(f" > Global Stack Size: {len(self.global_stack)}")
        while len(self.global_stack) > 0:
            self.print("STACK_TRACK: Popped Local Assumption Stack in process_unsat_core")
            local_assumption_stack = self.global_stack.pop()
            self.print(f" > Local Stack Size: {len(local_assumption_stack)}")
            while len(local_assumption_stack) > 0:
                atomic_assumption_stack = local_assumption_stack.pop()
                found = not atomic_assumption_stack.is_empty()
                if found:
                    break
            if found:
                break
        #if len(self.global_stack) > 0:
        #    local_assumption_stack = self.global_stack.pop()
        #if len(local_assumption_stack) > 0:
        #    atomic_assumption_stack = local_assumption_stack.pop() ## Remember that assumption stacks also include checkpoints
        #    ## Find the first non-empty atomic_assumption_stack
        #    while atomic_assumption_stack.is_empty():
        #        atomic_assumption_stack = local_assumption_stack.pop() ## Remember that assumption stacks also include checkpoints
        ## After this loop, the following statements are true:
        ## -- At least 1 local stack has been removed from the global stack
        ## -- At least 1 atomic assumption stack has been removed from the local stack
        ## -- The most recent faulty assumption is not contained in the global stack
        ## -- The most recent faulty assumption is not contained in the local stack
        
        ## Restore the table checkpoint to the corresponding atomic assumption stack
        self.print(" !!! Restoring Checkpoint ...")
        self.print(atomic_assumption_stack.checkpoint)
        atomic_assumption_stack.checkpoint.restore(self)

        ## We have identified the atomic assumption stack. Now we need to backtrack / revert via symbolic backtracking.
        ## NOTE: Important: When this function exits, it must be as if we just finished a symbolic_fill
        self.print(" !!! Performing Backtracking inside Process UNSAT Core (first stage)...")
        unknown_relations = deque()
        ## We also need to reinitialize the solver information
        self.solver.reset()
        ## Make sure we explicitly add the domain of the variables, as we weren't doing this before.
        for _, variable in self.context.items():
            self.solver.add(define_domain(variable, self.sigma_O))

        ## NOTE: If the table from the previous iteration included equality assumptions, then 
        ## the following will include those possibly incorrectly unified variables.
        ## Therefore, we need to ensure that we keep a non-unified table around.
        ## In particular, it is always safe to perform unification over known equivalent variables,
        ## but we need to ensure that we 
        self.solver.add(self.constraints["seq_eq"])
        self.solver.add(self.constraints["seq_ineq"])
        self.add_equivalence_classes_to_solver()

        ## Add in the existing global stack, and the existing local stack
        self.global_stack.add_assumptions_to_solver(self.solver)
        local_assumption_stack.add_assumptions_to_solver(self.solver)
        #self.solver.add(assertion) ## From UNSAT
        self.print(f" Symbolic Fill: Assumptions MUST satify:")
        self.print(self.solver)
        restart_loop = False
        ## Reinitialize the solver state
        for assumption in atomic_assumption_stack.assumption_stack:
            b, v1, v2 = assumption
            self.solver.push()
            self.solver.add(v1 == v2 if b else v1 != v2)
        self.print(f"Reinitialized assumptions from assumption stack:")
        self.print(self.solver)
        self.print(f"The Assumption Stack used was:")
        self.print(atomic_assumption_stack)

        self.print(f"Walking back the atomic assumption stack...")
        ## Perform backtracking using the atomic assumption stack -- here, we assumed that the assumption stack is non-zero.
        ## However, we also need to consider the case when the atomic assumption stack is EMPTY.
        while len(atomic_assumption_stack) > 0 and (not atomic_assumption_stack.last()[0]):
            if len(atomic_assumption_stack) == 1:
                self.print(f"Current atomic assumption stack has been fully explored; moving to previous atomic assumption stack")
                atomic_assumption_stack = self.symbolic_fill_backtrack(local_assumption_stack, atomic_assumption_stack, unknown_relations)
                restart_loop = True
                break

            b, vv1, vv2 = atomic_assumption_stack.pop()
            self.solver.pop()
            unknown_relations.append((not b, vv1, vv2))

        if not restart_loop and len(atomic_assumption_stack) > 0:
            b, vv1, vv2 = atomic_assumption_stack.pop()
            self.solver.pop()

            unknown_relations.append((not b, vv1, vv2))

        self.print(f"The Assumption stack is Now:")
        self.print(atomic_assumption_stack)
        self.print(f"The unknown_relations is now:")
        self.print(unknown_relations)
        self.print(f"The Solver is now:")
        self.print(self.solver)

        restart_loop = True       
        ## Now unknown_relations is initialized correctly 
        while restart_loop:
            restart_loop = False
            self.solver.reset()
            ## Make sure we explicitly add the domain of the variables, as we weren't doing this before.
            for _, variable in self.context.items():
                self.solver.add(define_domain(variable, self.sigma_O))
 
            ## NOTE: If the table from the previous iteration included equality assumptions, then 
            ## the following will include those possibly incorrectly unified variables.
            ## Therefore, we need to ensure that we keep a non-unified table around.
            ## In particular, it is always safe to perform unification over known equivalent variables,
            ## but we need to ensure that we also include the equivalence relations from the equivalence
            ## classes when providing info to the solvers. Otherwise, the solver will have missing constraints,
            ## since a substituted variable does NOT imply an equivalence relation.
            self.solver.add(self.constraints["seq_eq"])
            self.solver.add(self.constraints["seq_ineq"])
            self.add_equivalence_classes_to_solver()

            ## Add in the existing global stack, and the existing local stack
            self.global_stack.add_assumptions_to_solver(self.solver)
            local_assumption_stack.add_assumptions_to_solver(self.solver)
            #self.solver.add(assertion)
            ## Reinitialize the solver state
            for assumption in atomic_assumption_stack.assumption_stack:
                b, v1, v2 = assumption
                self.solver.push()
                self.solver.add(v1 == v2 if b else v1 != v2)
            self.print(f" Symbolic Fill: Assumptions MUST satify:")
            self.print(self.solver)

            ## NOTE: Step 4. Test a variety of assumptions. Using a stack, we can simulate a left-to-right binary decision tree
            ## traversal, where each node is a proposition (which represents an assumption), and the left and right edges
            ## indicate whether the node's proposition is true or false. The leftmost branch of the tree represents when
            ## all propositions are true, and the rightmost branch of the tree represents when all the propositions are
            ## false. When the leaves of the tree are reached, we perform a test indicating whether all assumptions along a path
            ## in the tree are compatible with the existing assumptions. In fact, we can perform this test whenever we reach a new
            ## node in the tree.
            while len(unknown_relations) > 0:
                relation, var1, var2 = unknown_relations.pop()

                atomic_assumption_stack.push((relation, var1, var2))
                self.solver.push()
                self.solver.add((var1 == var2) if relation else (var1 != var2))

                ## Check if a contradiction arises.
                ## NOTE: Important -- we must also ensure that these newly made assumptions do not conflict with
                ## any previously made assumptions.
                sat_unsat = self.solver.check()
                self.print(f" Testing Hypothesis: {self.solver}")
                ## If unsat, then our hypothesis is wrong.
                ## If too many equivalence classes, then our hypothesis is wrong.
                ## NOTE: The fails_representative_check only occurs if we are at a leaf node,
                ## and if so, performs a temporary unification to determine the total number of
                ## equivalence classes that would remain after unification.
                if sat_unsat != z3.sat:
                    fails_representative_check = True
                else:
                    fails_representative_check = self.check_representatives(unknown_relations, atomic_assumption_stack, local_assumption_stack)

                if sat_unsat != z3.sat or fails_representative_check:
                    self.print(" !!! Performing Backtracking inside Process UNSAT Core (second stage)")
                    while len(atomic_assumption_stack) > 0 and (not atomic_assumption_stack.last()[0]):
                        if len(atomic_assumption_stack) == 1:
                            atomic_assumption_stack = self.symbolic_fill_backtrack(local_assumption_stack, atomic_assumption_stack, unknown_relations)
                            restart_loop = True
                            break

                        b, vv1, vv2 = atomic_assumption_stack.pop()
                        self.solver.pop()
                        unknown_relations.append((not b, vv1, vv2))

                    if restart_loop:
                        break

                    if len(atomic_assumption_stack) > 0:
                        b, vv1, vv2 = atomic_assumption_stack.pop()
                        self.solver.pop()

                        unknown_relations.append((not b, vv1, vv2))
        
        ## Perform unification
        self.unification(atomic_assumption_stack, local_assumption_stack)
        ## Return
        return atomic_assumption_stack


    def backtrack_assumptions(self, unsat_core, input_stack, big_step=False):
        """
        This function enables backtracking of the assumption and hypothesis stacks.

        The purpose of BigStep backtrack is to identify the nearest LocalAssumptionStack
        that contains the hypothesis which does not violate constraints. Once this local
        assumption stack is identified, we can perform small step backtracking on it as normal.

        Under the BigStep strategy, we need to find a pair of consecutive hypotheses
        h_{k-1} and h_k such that h_{k-1} passes and h_k fails.

        To test each hypothesis, there are two options:
            (A) Validate using only constraints known at that point in time.
            (B) Alternatively, we can just use the non-unified constraints that are currently known, since there
                is guaranteed to be a solution
            However, in both cases, we only use assumptions available at that point in time.
        """
        def is_in(v, it):
            return z3.Or([v == r for r in it])

        local_assumption_stack = input_stack
        if big_step:
            self.print(" !!! Performing BIG_STEP backtracking")
            solver = z3.Solver()
            solver.reset()
            ## Recall that sequence preferences along with a hypothesis (closed and consistent symbolic observation table)
            ## yield constraints over variables in the table.
            ## 
            ## Together,
            ##   self.old_sorted (ordering over known sequence representatives)
            ##   constraints["seq_EC"] (equivalence classes of sequences)
            ## represent the total ordering over sequences which have been queried.
            ## 
            ## Constraints["seq_eq"] and constraints["seq_ineq"] represent particular instantiations from preferences
            ## to constraints over variables.
            ##
            ## To ensure correctness, we should generate the constraints from the known sequence preferences and
            ## the table under consideration, unless all constraints are kept in non-unified form.
            ## TODO [DONE]: Ensure that seq_eq and seq_ineq are kept in non-unified form.
            ## We assume the seq_eq and and seq_ineq are in non-unified format.
            solver.add(self.constraints["seq_eq"])
            solver.add(self.constraints["seq_ineq"])
            self.add_equivalence_classes_to_solver()

            ## TODO: Also, keep a version of non-unified representatives around
            for rep, val in self.constraints["repr"].items():
                if val is None:
                    solver.add( is_in(rep, self.sigma_O) )
                else:
                    solver.add( rep == val)
            ## Perform Big Step backtracking on the Global Stack.
            local_assumption_stack = self.global_stack.backtrack(solver, self.constraints)
        
        self.print(" !!! Performing SMALL_STEP backtracking on LocalAssumptionStack")
        atomic_assumption_stack = self.process_unsat_core(local_assumption_stack, unsat_core)
        local_assumption_stack.push(atomic_assumption_stack)
        return local_assumption_stack

        ## Now local_assumption_stack is the stack which contains the bad assumption.
        ## We perform regular backtracking here on the local assumption stack
        #atomic_assumption_stack = local_assumption_stack.backtrack()
        #self.restore_checkpoint(atomic_assumption_stack)

    def add_equivalence_classes_to_solver(self, negate=False, EC=None, solver=None):
        """
        Given a set of equivalence classes, update the solver to include that equality information.
        """
        if EC is None:
            EC = self.constraints["EC"]
        if solver is None:
            solver=self.solver
        reps = set()
        for k, v in EC.items():
            reps.add(v.repr())

        all_pairs = list()
        for rep in reps:
            ## Only add constraints for equivalence classes that have at least size 2
            if len(EC[rep]) > 1:
                all_pairs.extend(EC[rep].make_pairs())

        ## All constraint for all pairs, or NOT all pairs
        if not negate:
            solver.add(z3.And(all_pairs))
        else:
            ## NOTE: (|\Gamma| <= k --> !E) <==> (E --> |\Gamma| > k)
            ## This essentially enforces that if the dimensionality is less that or equal to k, 
            ## then this set of equivalence classes is not valid. This is equivalent to:
            ## If this set of equivalence classes is valid, then the dimensionality needs
            ## to be larger than k.
            solver.add(z3.Implies(self.G <= len(self.context), z3.Not(z3.And(all_pairs))))

    def add_cex_to_solver(self, hypothesis, cexs=None, solver=None):
        if cexs is None:
            cexs = self.constraints["seq_sym"]
        if solver is None:
            solver = self.solver

        if hypothesis is not None:
            for cex, concrete_val, boolean in cexs:
                induced_cex_constraint = self.generate_cex_constraint(cex, concrete_val, boolean, *hypothesis)
                solver.add(induced_cex_constraint)
        else:
            ## we don't have a hypothesis, so attempt to add
            ## all constraints based on variables in self.context
            for cex, concrete_val, boolean in cexs:
                if cex in self.context:
                    sym_acc = self.valuation_model(cex)
                    if boolean:
                        solver.add(sym_acc == concrete_val)
                    else:
                        solver.add(z3.Not(sym_acc == concrete_val))

    def restore_checkpoint(self, atomic_assumption_stack):
        atomic_assumption_stack.checkpoint.restore(self)
        self.print("RESTORED CHECKPOINT:")
        self.print(self)

    def push_local_assumption_stack(self, stack):
        ## Push the local assumption stack to the global stack
        self.global_stack.push(stack)

    def push_checkpoint(self):
        ## Create a new checkpoint
        checkpoint = Checkpoint(self)
        self.checkpoint_stack.append(checkpoint)

    def pop_checkpoint(self):
        if len(self.checkpoint_stack) > 0:
            checkpoint = self.checkpoint_stack.pop()
            checkpoint.restore(self)

    def push_model(self, m):
        self.model_stack.append(m)

    def pop_model(self):
        if len(self.model_stack) > 0:
            return self.model_stack.pop()
        else:
            return None

    def last_model(self):
        return self.model_stack[-1]

    def build_seq2idx(self):
        seq2idx = dict()
        idx = 0
        EC = self.constraints["seq_EC"]
        RP = self.constraints["seq_repr"]
        for el in self.old_sorted:
            if el not in EC:
                EC[el] = EquivalenceClass(set((el,)))
                rep = EC[el].repr()
                RP[rep] = None

            for member in EC[el].members:
                seq2idx[member] = idx
            idx += 1
        return seq2idx

    def __query_total_order(self, seq2idx, A, B):
        """
        Returns:
           1 if T(A) > T(B)
          -1 if T(A) < T(B)
           0 if T(A) = T(B)
        None if nothing can be concluded.
        """
        ## Internal sequence decomposition function
        def decompose(a1, b1):
            a = a1[:len(a1)-1]
            x = a1[len(a1)-1:]
            b = b1[:len(b1)-1]
            y = b1[len(b1)-1:]
            return a, x, b, y

        ## Internal lookup function
        def prefQ(m, p, q):
            ## TODO: Handle looking up representative
            return (m[p] > m[q]) - (m[p] < m[q])

        ## We record the total ordering in seq2idx. We assume that seq2idx includes only Value Representatives.
        ## Given a pair sequences p = ax and q = by, with len(x) = 1, len(y) = 1, len(a) >= 0, and len(b) >= 0,
        ## we perform a test that determines whether:
        ## (a) T(p) > T(q), (b) T(p) < T(q), (c) T(p) == T(q), or (d) inconclusive
        ## based on a decision tree involving:
        ## 1. prefQ(a, b), prefQ(ax, by)
        ## 2. prefQ(by,b), prefQ(ax, a)
        a, x, b, y = decompose(A, B)

        ## Level 1:
        X, Y = prefQ(seq2idx, A, B), prefQ(seq2idx, a, b)
        if abs(X+Y) == 2:
            ## Go to level 2:
            X, Y = prefQ(seq2idx, A, a), prefQ(seq2idx, B, b)
            if abs(X+Y) == 2:
                return None
            else:
                return (X > Y) - (X < Y)
        else:
            return (X > Y) - (X < Y)

    def quicksort(self, L):
        num_pref_queries = 0
        if len(L) <= 1:
            return L, num_pref_queries

        ## Pick a random pidx for the pivot
        pidx = random.randint(0,len(L)-1)
        p2 = L[pidx]

        L_list = list()
        R_list = list()

        for idx in range(len(L)):
            if idx == pidx:
                continue
            p1 = L[idx]
            p = self.preference_query(p1, p2)
            self.update_constraint_set_value(p, p1, p2) ## Send the pair to the contraint set
            num_pref_queries += 1

            if p < 0:
                L_list.append(p1)
            elif p > 0:
                R_list.append(p1)
            else: ## Equal, so don't append anything to either list
                pass ## Creating the equivalence class can be deferred to the unification step

        sorted_L_list, nl_prefs = self.quicksort(L_list)
        sorted_R_list, nr_prefs = self.quicksort(R_list)

        num_pref_queries += nl_prefs
        num_pref_queries += nr_prefs

        sorted_list = list()
        sorted_list.extend(sorted_L_list)
        sorted_list.append(p2)
        sorted_list.extend(sorted_R_list)
        return sorted_list, num_pref_queries

    ## DONE 
    ## [TESTED][DONE]
    def unification(self, equivalence_classes):
        """
        With the SMT model, we have already identified the equivalence classes for this round
        Simply perform the substitution in the table. Leave the constraints alone.
        NOTE: We need an easy way to identify all the equality constraints.

        We will consider each individual constraint as an expression; therefore we will need to be able to identify the
        individual terms in the each expression, and if necessary, modify the expression via substitution.

        We will also need to determine whether an expression contains a particular term.
        """
        ## Step 1: Identify and construct equivalence classes based on the equality constraints,
        ## and establish representatives of each equivalence class.
        ## NOTE: Since we are creating equivalence classes for variables, we don't need to keep
        ## the original equality constraints around (they are encoded into the equivalence classes)
        ## Therefore, we will always just encode the equality constraints into equivalence classes,
        ## and just keep the equivalence classes around. If two variables belong to the same equivalence
        ## class, then they have an implicit equality between them.
        ## NOTE: We also need to take care of the cases where there are no equalities available. In that
        ## case, individual variables go into their own equivalence classes.
        self.print(f"=== >>> UNIFICATION STEP <<< ===\n")

        ## NOTE: We need to perform a selective unification strategy:
        ##       - Items that are SAFE to unify without affecting the outcome:
        ##          + KNOWN equality constraints which are deducible from ground truth constraints
        ##          KNOWN_EQ can be unified into SYMB_OBS_TABLE and INEQ without any issues.
        ##          + Representatives (REPR) should be only in terms of variables from KNOWN_EQ
        ##       - Items that are UNSAFE to unify:
        ##          + ATOMIC_ASSUMPTIONS should be not unified. They should be kept SEPARATE from
        ##            the KNOWN_EQs and KNOWN_INEQs.
        ##          + We add ATOMIC_ASSUMPTIONS as TRACKED_ASSERTIONS in the SMT solver.
        ## Therefore, our UNIFICATION algorithm will become a selective one. The procedure should
        ## occur as follows:
        ## - KNOWN_EQ: self.constraints["eq"] -> create equivalence classes -> representatives are in terms
        ##   of these known variables.
        ## - KNOWN_INEQ: self.constraints["ineq"] -> substitution -- express KNOWN_INEQ in terms of KNOWN_EQ
        ##   representatives
        ## - Perform substitution in the SYMB_OBS_TABLE using KNOWN_EQ.
        ## - Next, perform substitution in SEQ_EQ and SEQ_INEQ using KNOWN_EQ
        ## - Perform substitution in SYMB_OBS_TABLE using ATOMIC_ASSUMPTION_STACK (this can always be undone
        ##   since the previous table is stored in the ATOMIC_ASSUMPTION_STACK checkpoint).

        ## For unification on the symbolic observation table, we need to
        ## make a copy of EC and RP, then update those copies with the
        ## atomic assumption stack.

        try:
            ## Step 3: Perform substitution for each entry in the symbolic observation table
            nrows, ncols = self.table_upper.shape()
            self.print(f" PROCESSING UPPER TABLE ENTRIES: SHAPE is ({nrows}, {ncols})")
            for r in range(nrows):
                for c in range(ncols):
                    cur_var = self.table_upper.get_entry(r,c)
                    sub_var = equivalence_classes[cur_var].repr()
                    self.table_upper.set_entry(r,c, sub_var)
            
            nrows, ncols = self.table_lower.shape()
            self.print(f" PROCESSING LOWER TABLE ENTRIES: SHAPE is ({nrows}, {ncols})")
            for r in range(nrows):
                for c in range(ncols):
                    cur_var = self.table_lower.get_entry(r,c)
                    sub_var = equivalence_classes[cur_var].repr()
                    self.table_lower.set_entry(r,c, sub_var)

            self.print(f"UNIQUE TABLE VARIABLES:\n  {self.table_upper.get_entry_set() | self.table_lower.get_entry_set()}")
        except KeyError as err:
            self.print(f"UNIQUE TABLE VARIABLES:\n  {self.table_upper.get_entry_set() | self.table_lower.get_entry_set()}")
            self.print(f"{err}")
            exit()
        ## Step 4. Cache the equivalence classes for future use
        self.constraints["EC"] = equivalence_classes

    def solver_nonunified_constraints(self, hypothesis, solver=None):
        """
        Adds non-unified constraints to the solver

        Here, we define the semantics of the different constraint sets:

        We let
            "eq"
            "ineq"
            "EC"
            "repr"

        refer to equality, inequality, equivalence classes, and reprensentatives with respect to variables.
        These are derived from the seq_* constraints.

        We let
            "seq_eq"
            "seq_ineq"
            "seq_EC"
            "seq_repr"
            "seq_sym"

        refer to equality, inequality, equivalence classes, and representatives with respect to values, as represented
        symbolically by sequences.

        """
        def is_in(v, it):
            return z3.Or([v == r for r in it])

        this_solver = self.solver
        if solver is not None:
            this_solver = solver

        this_solver.reset()
        ## These are all the gathered known ground truth constraints from the teacher.
        ## Step 1: Add the constraints obtained from the Preference Queries
        this_solver.add(self.constraints["seq_eq"])
        this_solver.add(self.constraints["seq_ineq"])
        self.add_equivalence_classes_to_solver(EC=self.constraints["EC"], solver=this_solver)
        ## Step 2: Add the constraints obtained from the Equivalence Queries (the counterexamples)
        ## The current hypothesis must be used here: Given a hypothesis and counterexample, we can generate the constraint
        for cex, concrete_val, boolean in self.constraints["seq_sym"]:
            induced_cex_constraint = self.generate_cex_constraint(cex, concrete_val, boolean, *hypothesis)
            this_solver.add(induced_cex_constraint)

        ## Add all the variables in the context so far:
        ## Ensure the domain -- though, there might be a more straightforward way to do this..
        for rep, val in self.constraints["repr"].items():
            if val is None:
                this_solver.add( is_in(rep, self.sigma_O) )
            else:
                this_solver.add( rep == val)

    def solver_check_assumption_stack(self, solver=None, local_assumption_stack=None):
        """
        Add all assumptions that have been made up to this point in time for this particular hypothesis
        NOTE: Need to make sure duplicate assumptions are not transferred to the solver, since at certain
        times the global stack contains a copy of the local stack
        """
        this_solver = self.solver
        if solver is not None:
            this_solver = solver

        self.print(" !!! Adding Global Stack Assumptions")
        this_solver.push()

        ## Handle duplicate local assumption stack in global assumption stack if it exists
        last = None
        if len(self.global_stack) > 0:
            last = self.global_stack.pop()

        self.global_stack.track_assumptions_in_solver(this_solver)
        if local_assumption_stack is not None:
            self.print(" !!!!! Adding extra local assumptions")
            local_assumption_stack.track_assumptions_in_solver(this_solver)

        ## Handle duplicate local assumption stack in global assumption stack if it exists
        if last is not None:
            self.global_stack.push(last)

        self.print(this_solver)
        sat_unsat = this_solver.check()
        return sat_unsat

    def set_variable_value(self, seq, val):
        """
        Given a sequence, look up its corresponding variable, and add a constraint stating that the
        variable must be equal to a certain value

        NOTE: Our context should include our sequence
        """
        representative = self.constraints["EC"][self.context[seq]].repr()
        self.constraints["repr"][representative] = val
        self.print(f"SET VARIABLE-> VALUE:\n  SEQ: {seq}\n   VAR: {representative} == VAL {val}\n")


    def update_constraint_set_variable(self, p, p1, p2):
        """
        NOTE: Variables are shared between the upper and lower tables using the context, so each unique
        sequence corresponds to its own unique variable.

        This means all we have to do is look up the corresponding variable in the context using the sequence.
        That way, we don't have to deal with the upper/lower table prefix/suffix indexing.
        """
        if p == 0:
            self.constraints["eq"].add(self.context[p1] == self.context[p2])
        elif p < 0:
            self.constraints["ineq"].add(self.context[p1] < self.context[p2])
        else:
            self.constraints["ineq"].add(self.context[p1] > self.context[p2])
    
    def update_constraint_set_value(self, p, p1, p2):
        """
        We generate a constraint based on the value function of the sequence, rather than the variable.

        Here, value is the summation function over prefixes

        Make it iterative -- add the constraints directly to the solver
        """

        ## Constraint updates
        try:
            v1 = self.valuation_model(p1)
            v2 = self.valuation_model(p2)
            # v1 = z3.Sum(tuple(self.context[x] for x in range_prefixes(p1)))
            # v2 = z3.Sum(tuple(self.context[x] for x in range_prefixes(p2)))
        except KeyError as err:
            self.print(self)
            self.print("Context")
            self.print(self.context)
            self.print("p1")
            self.print(p1)
            self.print("p2")
            self.print(p2)
            raise err

        if p == 0:
            self.constraints["seq_eq"].add(v1 == v2)
            ## NOTE: Here, we also need to construct the sequence equivalence classes (wrt value)
            self.update_seq_ECs(p1, p2)
            self.solver.add(v1 == v2)
        elif p < 0:
            self.constraints["seq_ineq"].add(v1 < v2)
            self.solver.add(v1 < v2)
        else:
            self.constraints["seq_ineq"].add(v1 > v2)
            self.solver.add(v1 > v2)

    def update_seq_ECs(self, p1, p2):
        """
        Handles merging equivalence classes over sequences
        """
        EC = self.constraints["seq_EC"] ## Dictionary of equivalence classes -- Sequence -> EquivalenceClass
        RP = self.constraints["seq_repr"] ## Dictionary of representatives -- Sequence -> Value

        L = p1
        R = p2
        if L in EC and R in EC:
            ## Check if we need to merge equivalence classes
            if not (EC[L] is EC[R]):
                ## NOTE: We only keep the "left" representative
                R_rep = EC[R].repr()
                L_rep = EC[L].repr()
                ## Add all Right EC class members to the left
                EC[L].update(EC[R])
                ## Make all Right EC class members refer to the left EC class
                R_EC_members = EC[R].members
                for member in R_EC_members:
                    EC[member] = EC[L]
                if not bool(L_rep == R_rep):
                    ## Preserve known values:
                    if RP[R_rep] is not None:
                        RP[L_rep] = RP[R_rep]
                    if R_rep in RP:
                        del RP[R_rep] ## If they have different representives, then delete the right one
            ## Else they already refer to the same equivalence class
        elif L in EC:
            EC[L].add(R)
            EC[R] = EC[L]
        elif R in EC:
            EC[R].add(L)
            EC[L] = EC[R]
        else:
            EC[L] = EquivalenceClass(set((L,R)))
            EC[R] = EC[L]
            rep = EC[L].repr()
            RP[rep] = None

    ## [DEPRECATED] Here for reference only.
    def __deprecated__update_constraint_set(self, p, p1, p2):
        """
        Need to be careful about making sure we refer to the same set of variables
        between the upper and lower tables.
        Also, given a prefix, how do we identify if it came from the upper or lower table?
            - It is possible for the same prefix to exist in both sections of the table
            - Simply add redundant constraints; we'll remove them later during the unification process...

        NOTE: Variables are shared between the upper and lower tables using the context, so each unique
        sequence corresponds to its own unique variable.

        This means all we have to do is look up the corresponding variable in the context using the sequence.
        That way, we don't have to deal with the upper/lower table prefix/suffix indexing.
        """
        s1, e1 = p1
        s2, e2 = p2
        c1, c2 = -1
        c1 = self.suffix_set.get(e1)
        c2 = self.suffix_set.get(e2)
        if p == 0:
            r1, r2 = -1
            if s1 in self.prefix_set:
                r1 = self.prefix_set.get(s1)
                if s2 in self.prefix_set:
                    r2 = self.prefix_set.get(s2)
                    self.constraints["eq"].add(self.table_upper.get_entry(r1, c1) == self.table_upper.get_entry(r2, c2))
                if s2 in self.prefix_alpha_set:
                    r2 = self.prefix_alpha_set.get(s2)
                    self.constraints["eq"].add(self.table_upper.get_entry(r1, c1) == self.table_lower.get_entry(r2, c2))
            if s1 in self.prefix_alpha_set:
                r1 = self.prefix_set.get(s1)
                if s2 in self.prefix_set:
                    r2 = self.prefix_set.get(s2)
                    self.constraints["eq"].add(self.table_lower.get_entry(r1, c1) == self.table_upper.get_entry(r2, c2))
                if s2 in self.prefix_alpha_set:
                    r2 = self.prefix_alpha_set.get(s2)
                    self.constraints["eq"].add(self.table_lower.get_entry(r1, c1) == self.table_lower.get_entry(r2, c2))
        elif p < 0:
            r1, r2 = -1
            if s1 in self.prefix_set:
                r1 = self.prefix_set.get(s1)
                if s2 in self.prefix_set:
                    r2 = self.prefix_set.get(s2)
                    self.constraints["ineq"].add(self.table_upper.get_entry(r1, c1) < self.table_upper.get_entry(r2, c2))
                if s2 in self.prefix_alpha_set:
                    r2 = self.prefix_alpha_set.get(s2)
                    self.constraints["ineq"].add(self.table_upper.get_entry(r1, c1) < self.table_lower.get_entry(r2, c2))
            if s1 in self.prefix_alpha_set:
                r1 = self.prefix_set.get(s1)
                if s2 in self.prefix_set:
                    r2 = self.prefix_set.get(s2)
                    self.constraints["ineq"].add(self.table_lower.get_entry(r1, c1) < self.table_upper.get_entry(r2, c2))
                if s2 in self.prefix_alpha_set:
                    r2 = self.prefix_alpha_set.get(s2)
                    self.constraints["ineq"].add(self.table_lower.get_entry(r1, c1) < self.table_lower.get_entry(r2, c2))
        else:
            r1, r2 = -1
            if s1 in self.prefix_set:
                r1 = self.prefix_set.get(s1)
                if s2 in self.prefix_set:
                    r2 = self.prefix_set.get(s2)
                    self.constraints["ineq"].add(self.table_upper.get_entry(r1, c1) > self.table_upper.get_entry(r2, c2))
                if s2 in self.prefix_alpha_set:
                    r2 = self.prefix_alpha_set.get(s2)
                    self.constraints["ineq"].add(self.table_upper.get_entry(r1, c1) > self.table_lower.get_entry(r2, c2))
            if s1 in self.prefix_alpha_set:
                r1 = self.prefix_set.get(s1)
                if s2 in self.prefix_set:
                    r2 = self.prefix_set.get(s2)
                    self.constraints["ineq"].add(self.table_lower.get_entry(r1, c1) > self.table_upper.get_entry(r2, c2))
                if s2 in self.prefix_alpha_set:
                    r2 = self.prefix_alpha_set.get(s2)
                    self.constraints["ineq"].add(self.table_lower.get_entry(r1, c1) > self.table_lower.get_entry(r2, c2))
        

def symbolic_lstar(input_alphabet, output_alphabet, teacher, valuation_model, gamma, use_cc_obj=True, use_ve_obj=True, use_cex_expansion=False, use_ids=True):
    """
    Here, we implement the symbolic lstar algorithm, using preferences and I/O examples

    NOTE: For the empty sequence, we state that the output is set to some value.
    NOTE: gamma is now represented as an exact rational value (using fractions.Fraction)
    """
    initial_constraints = (tuple(), fractions.Fraction(0), True)
    if valuation_model == "prod":
        initial_constraints = (tuple(), fractions.Fraction(1), True)
    seq_sym = set()
    seq_sym.add(initial_constraints)

    print(f"INITIALIZATION:\n")
    sym_obs_table = SymbolicObservationTable(
        input_alphabet, output_alphabet, teacher.preference_query, teacher.equivalence_query,
        prefix = None, suffix = None,
        constraints = {
                "eq": set(), "ineq": set(), "EC": dict(), "repr": dict(),
                "seq_eq": set(), "seq_ineq": set(), "seq_EC": dict(), "seq_repr": dict(),
                "seq_sym": seq_sym,
        },
        context = dict(),
        valuation = valuation_model,
        gamma_val = gamma,
        use_closed_and_consistent_obj = use_cc_obj,
        use_variable_equivalence_obj = use_ve_obj,
        use_ids=use_ids)

    initial_constraint = (tuple(), fractions.Fraction(0))
    if valuation_model == "prod":
        initial_constraint = (tuple(), fractions.Fraction(1))
    
    print(sym_obs_table)

    local_assumption_stack = LocalAssumptionStack()
    force_expansion = sym_obs_table.symbolic_fill(init_constraint=initial_constraint)
    is_correct = False
    is_consistent = False
    is_closed = False
    hypothesis = None
    counterexamples = list()

    sym_obs_table.record_event("initialization")
    while not is_correct:

        ## Step 1: Get a valid hypothesis
        hypothesis = None
        h_solver = None
        satisfied = False
        while not satisfied:

            is_consistent, alpha_suffix = sym_obs_table.is_consistent()
            is_closed, prefix_alpha = sym_obs_table.is_closed()

            while not (is_closed and is_consistent):
                if not is_consistent:
                    sym_obs_table.expand_suffixes(alpha_suffix)
                    force_expansion = sym_obs_table.symbolic_fill()
                    ## Event recording must come AFTER the symbolic fill
                    sym_obs_table.record_event("consistency")

                while force_expansion:
                    sym_obs_table.forced_expansion(counterexamples)
                    force_expansion = sym_obs_table.symbolic_fill()
                    sym_obs_table.record_event("expansion")
            
                is_consistent, alpha_suffix = sym_obs_table.is_consistent()
                is_closed, prefix_alpha = sym_obs_table.is_closed()

                if not is_closed:
                    sym_obs_table.expand_prefixes(prefix_alpha)
                    force_expansion = sym_obs_table.symbolic_fill()
                    ## Event recording must come AFTER the symbolic fill
                    sym_obs_table.record_event("closure")
                
                while force_expansion:
                    sym_obs_table.forced_expansion(counterexamples)
                    force_expansion = sym_obs_table.symbolic_fill()
                    sym_obs_table.record_event("expansion")

                #sym_obs_table.symbolic_fill()
                is_consistent, alpha_suffix = sym_obs_table.is_consistent()
                is_closed, prefix_alpha = sym_obs_table.is_closed()

            sat_unsat, hypothesis = sym_obs_table.make_hypothesis()
            ## If we get a previous symbolic hypothesis, then we need to backtrack. Either the solution is contained in the symbolic
            ## hypothesis, or it is not; therefore any previously tested symbolic hypotheses cannot be correct.
            ## We should probably also enforce that subsequent hypotheses should not be too distant from the previous hypothesis. For
            ## example, don't go from 3 states to 10 states unless 4 though 9 states have been eliminated.
            if sat_unsat == z3.unsat:
                sym_obs_table.print(" >>>=== !!! Unsatisfiable Constraints -- cannot construct concrete hypothesis !!! ===<<<\n")
                ## [DONE]: The current equivalence class assumption is invalid. This means
                ## we need to add the negated equivalence class and pop the current table
                ## We can do this through running a modified version of symbolic_fill
                sym_obs_table.symbolic_fill(backtrack=True)
                satistied = False ## We will continue back at the beginning of the while loop
            else:
                satisfied = True ## We will exit the while loop so that we can test the concrete hypothesis
        local_assumption_stack.set_hypothesis(hypothesis)

        ## Step 2: Check if a counterexample exists for the hypothesis.
        ## NOTE: There is a possibility that multiple plausible solutions for the representatives exist for a given hypothesis. This
        ## means that when a counterexample is returned and the extra constraint added, then it is possible for more than one counter
        ## example to be returned in this loop. In that case, we need to make sure to add ALL the prefixes for all counterexamples to
        ## the table, otherwise, there is a possibility that we will miss expanding the upper table when it needs to be expanded. An 
        ## example is the following: suppose b was already in the upper table. If we get bb then b as counterexamples, if we only kept
        ## the last counterexample b, then bb would never be added to the table.
        try_again = True
        sym_obs_table.solver.push() ## This push level is to keep track of concrete hypotheses
        while try_again:
            ## Unpack the hypothesis for the query
            is_correct, result = sym_obs_table.equivalence_query(*hypothesis)
            sym_obs_table.record_event("equivalence")
            if not is_correct:
                cex, val, is_strong = result
                print(f"The value of VAL is {val}")
                print(f"The type of VAL is {type(val)}")
                print(f"VAL to string is {str(val)}")
                sym_obs_table.record_cex_length(cex)
                ## Add V(cex) = val (or V(cex) != hypothesis(cex))
                ## We encode the constraint cex == val as (cex,val,True)
                ## We encode the constraint cex != val as (cex,val,False)
                symbolic_val, concrete_val = sym_obs_table.symbolic_eval(cex, *hypothesis)
                if is_strong:
                    sym_obs_table.constraints["seq_sym"].add((cex, fractions.Fraction(val), is_strong))
                else:
                    sym_obs_table.constraints["seq_sym"].add((cex, fractions.Fraction(concrete_val), is_strong))
                counterexamples.append(cex)

                ## Attempt to make a new concrete hypothesis
                sat_unsat, hypothesis = sym_obs_table.new_concrete_hypothesis(*hypothesis)
                
                ## NOTE: We will need to analyze WHY the teacher provided this counterexample:
                ## We can add the concrete CEX to the solver, then analyze whether under this
                ## constraint if there are still solutions or not:
                ## 1. If SAT, then there exists some other valid concrete solution under the current symbolic
                ##    hypothesis. Therefore, try another concrete hypothesis and try again.
                ## 2. If UNSAT, then there are two cases we need to consider:
                ##    (A) If the UNSAT core is EMPTY, then the assumptions in our assumption stack are correct.
                ##        Therefore, there are currently no concrete solutions, so we need to proceed to processing
                ##        the counterexample by adding it and all its prefixes to the prefix set.
                ##    (B) If the UNSAT core is NON-EMPTY, then something in our assumption stack is incorrect.
                ##        This means that we need to backtrack on the stack. We can do this is a big step fashion

                ## TODO: Make sure the sym_obs_table solver uses non-unified constraints
                if sat_unsat == z3.sat:
                    try_again = True
                    ## Replace the old hypothesis with the new hypothesis in the local stack
                    sym_obs_table.print("Overwriting old concrete hypothesis with new concrete hypothesis...")
                    local_assumption_stack.set_hypothesis(hypothesis)
                    ## Then an equivalence query will be tested again
                else: ## UNSAT
                    sym_obs_table.print("Addition of the CEX implies that the existing constraints are now UNSAT...")
                    try_again = False
            else:
                ## If we get here, this implies that the hypothesis we found is correct
                sym_obs_table.print("Hypothesis is correct...")
                try_again = False
        sym_obs_table.solver.pop() ## Pop the concrete hypothesis cex constraints, since they are only valid for this symb hypothesis
        ## Step 3: If a counterexample was found, identify WHY it is a counterexample
        ## If we obtained a counterexample, we need to analyze for why is was UNSAT
        if not is_correct:
            forced_expansion = sym_obs_table.symbolic_fill(backtrack=True)
            if use_cex_expansion:
                ## Expand the table here:
                for cex in counterexamples:
                    for cex_prefix in range_prefixes(cex):
                        sym_obs_table.expand_prefixes(cex_prefix, expand_table=False)
                sym_obs_table.record_event("cex-expansion")
                force_expansion = sym_obs_table.symbolic_fill()

            while force_expansion:
                sym_obs_table.forced_expansion(counterexamples)
                force_expansion = sym_obs_table.symbolic_fill()
                sym_obs_table.record_event("expansion")
            
    return hypothesis, sym_obs_table.experimental_data()
