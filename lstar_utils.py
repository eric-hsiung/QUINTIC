import z3
from datastructures import EquivalenceClass
import fractions
class SymbolicHypothesis:
    """
    A symbolic hypothesis automaton structure
    """
    def __init__(self, q0, Q, delta, sigma):
        """
        Initial State : q0
        States : Q
        Transition Function : delta
        """
        self.q0 = q0
        self.Q = Q ## Dict: row -> prefix
        self.delta = delta ## Dict:  row x letter -> row
        self.sigma = sigma

    def __len__(self):
        return len(self.Q)

    def __eq__(self, other):
        """
        Checks if this hypothesis is equivalent to the other hypothesis
        """
        if len(self) != len(other) or self.sigma != other.sigma:
            return False

        equivalences = {self.q0 : other.q0}
        unprocessed = [self.q0]
        while len(unprocessed) > 0:
            a = unprocessed.pop()
            b = equivalenes[a]
            for c in self.sigma:
                ac = self.delta[a][c]
                bc = other.delta[b][c]
                if ac not in equivalences:
                    equivalences[ac] = bc
                    unprocessed.append(ac)
                elif equivalences[ac] != bc:
                    return False
        return True

def assert_rows_equal(row1, row2):
    return z3.And([row1[i] == row2[i] for i in range(len(row1))])

def define_domain(v, it):
    return z3.Or([v == r for r in it])

## Add utility function for extracting z3 variables from formula
## [TESTED][DONE]
def concat(first, second, as_type=tuple):
    if first is None or second is None:
        if first is None and second is None:
            return as_type()

        if first is None and second is not None:
            if isinstance(second, as_type):
                return second
            else:
                return as_type(second)

        if first is not None and second is None:
            if isinstance(first, as_type):
                return first
            else:
                return as_type(first)

    ## Both are not None
    a = None
    b = None
    if isinstance(first, (tuple, list, str)):
        a = as_type(first)
    else:
        a = as_type((first,))

    if isinstance(second, (tuple, list)):
        b = as_type(second)
    else:
        b = as_type((second,))

    return a + b

## TEST COVERAGE: [COMPLETE]
## [TESTED][DONE]
def get_vars(f):
    """
    Given a formula, return the leaves in the AST tree
    NOTES:
        f.decl() -- gives the root operator
        f.children() -- list of children
        f.num_args() -- number of children in the list
        f.arg(0), f.arg(1) -- left and right subtrees
    """
    r = set()
    def collect(f):
        ## Base case -- f is a leaf, which means it has no children:
        if f.num_args() == 0:
            if not isinstance(f, (z3.z3.IntNumRef)):
                r.add(f) ## Only add variables; exclude numerical constants
        else: ## f has children
            for c in f.children():
                collect(c)
    collect(f)
    return r

## TEST COVERAGE: [COMPLETE]
## [TESTED][DONE]
def range_prefixes(seq):
    """
    Yields all prefixes of a sequence
    """
    max_len = len(seq)
    for idx in range(max_len):
        yield seq[:idx+1]

class ValuationModel:
    def __init__(self, model_type, gamma_var=None, gamma_val=None, tolerance=1e-6):

        #assert gamma_val is None or isinstance(gamma_val, fractions.Fraction)

        def __summation_generate_cex_constraint__(seq, val, boolean, states, sigma_I, sigma_O, init_state, delta, output_fnc):
            """
            Given a counterexample constraint V(seq) != val, and a hypothesis h, generates the symbolic
            constraint induced by h.

            NOTE: We must explicitly include the initial state
            """
            q = init_state
            var_list = []
            var_list.append(q[0])
            for a in seq:
                q = delta[q][a]
                var_list.append(q[0])
            accumulation = z3.Sum(var_list)
            return z3.Not(accumulation == val) if not boolean else z3.Or(accumulation == val)

        def __discounted_summation_generate_cex_constraint__(seq, val, boolean, states, sigma_I, sigma_O, init_state, delta, output_fnc):
            """
            Given a counterexample constraint V(seq) != val, and a hypothesis h, generates the symbolic
            constraint induced by h.

            NOTE: We must explicitly include the initial state
            """
            #assert isinstance(val, fractions.Fraction), (val, type(val), gamma_val, type(gamma_val))

            q = init_state
            var_list = []
            var_list.append(q[0])
            n = 0
            for a in seq:
                n = n + 1
                q = delta[q][a]
                var_list.append(q[0]*(gamma_var**n))
            accumulation = z3.Sum(var_list)
            ## NOTE: If strong feedback, then use tolerance to generate the constraint
            #tolerance_expression = z3.And(accumulation < (fractions.Fraction(val) + fractions.Fraction(1,1000000)), (fractions.Fraction(val) - fractions.Fraction(1,1000000)) < accumulation)
            if not boolean:
                #return z3.Not(tolerance_expression)
                return z3.Not(accumulation == val)
            else:
                #return tolerance_expression 
                return z3.Or(accumulation == val) ## Swap to exact rational representation allows us to eliminate tolerance
                #return z3.And(accumulation < (val + tolerance), (val - tolerance) < accumulation)
        
        def __product_generate_cex_constraint__(seq, val, boolean, states, sigma_I, sigma_O, init_state, delta, output_fnc):
            """
            Given a counterexample constraint V(seq) != val, and a hypothesis h, generates the symbolic
            constraint induced by h.

            NOTE: We must explicitly include the initial state
            """
            q = init_state
            var_list = []
            var_list.append(q[0])
            for a in seq:
                q = delta[q][a]
                var_list.append(q[0])
            accumulation = z3.Product(var_list)
            return z3.Not(accumulation == val) if not boolean else z3.Or(accumulation == val)
        
        def __classification_generate_cex_constraint__(seq, val, boolean, states, sigma_I, sigma_O, init_state, delta, output_fnc):
            """
            Given a counterexample constraint V(seq) != val, and a hypothesis h, generates the symbolic
            constraint induced by h.

            NOTE: We must explicitly include the initial state
            """
            q = init_state
            var_list = []
            var_list.append(q[0])
            for a in seq:
                q = delta[q][a]
                var_list.append(q[0])
            accumulation = var_list[-1]
            return z3.Not(accumulation == val) if not boolean else z3.Or(accumulation == val)
        
        models = {
            "sum": __summation_generate_cex_constraint__,
            "discountsum": __discounted_summation_generate_cex_constraint__,
            "prod": __product_generate_cex_constraint__,
            "classification": __classification_generate_cex_constraint__,
        }

        def __summation_symbolic_eval__(seq, states, sigma_I, sigma_O, init_state, delta, output_fnc):
            """
            Evaluates the symbolic and concrete hypothesis simultaneously to obtain the symbolic
            and concrete values of a sequence

            NOTE: We must explicitly include the initial state, even though it is 0. This is to take care
            of the empty sequence.
            """
            q = init_state
            total_sum = 0
            var_list = []
            total_sum += output_fnc[q]
            var_list.append(q[0])
            for a in seq:
                q = delta[q][a]
                var_list.append(q[0])
                total_sum += output_fnc[q]

            symbolic_sum = z3.Sum(var_list)
            return symbolic_sum, total_sum

        def __discounted_summation_symbolic_eval__(seq, states, sigma_I, sigma_O, init_state, delta, output_fnc):
            """
            Evaluates the symbolic and concrete hypothesis simultaneously to obtain the symbolic
            and concrete values of a sequence

            NOTE: We must explicitly include the initial state, even though it is 0. This is to take care
            of the empty sequence.
            """
            q = init_state
            total_sum = fractions.Fraction(0)
            var_list = []
            total_sum += fractions.Fraction(output_fnc[q])
            var_list.append(q[0])
            n = 0
            for a in seq:
                n = n + 1
                q = delta[q][a]
                var_list.append(q[0]*(gamma_var**n))
                total_sum += fractions.Fraction(output_fnc[q])*(gamma_val**n)

            symbolic_sum = z3.Sum(var_list)
            return symbolic_sum, total_sum
        
        def __product_symbolic_eval__(seq, states, sigma_I, sigma_O, init_state, delta, output_fnc):
            """
            Evaluates the symbolic and concrete hypothesis simultaneously to obtain the symbolic
            and concrete values of a sequence

            NOTE: We must explicitly include the initial state, even though it is 0. This is to take care
            of the empty sequence.
            """
            q = init_state
            total_sum = 1
            var_list = []
            total_sum *= output_fnc[q]
            var_list.append(q[0])
            for a in seq:
                q = delta[q][a]
                var_list.append(q[0])
                total_sum *= output_fnc[q]

            symbolic_sum = z3.Product(var_list)
            return symbolic_sum, total_sum
        
        def __classification_symbolic_eval__(seq, states, sigma_I, sigma_O, init_state, delta, output_fnc):
            """
            Evaluates the symbolic and concrete hypothesis simultaneously to obtain the symbolic
            and concrete values of a sequence

            NOTE: We must explicitly include the initial state, even though it is 0. This is to take care
            of the empty sequence.
            """
            q = init_state
            total_sum = 0
            var_list = []
            total_sum = output_fnc[q]
            var_list.append(q[0])
            for a in seq:
                q = delta[q][a]
                var_list.append(q[0])
                total_sum = output_fnc[q]

            symbolic_sum = var_list[-1]
            return symbolic_sum, total_sum

        symbolic_evaluation_models = {
            "sum": __summation_symbolic_eval__,
            "discountsum": __discounted_summation_symbolic_eval__,
            "prod": __product_symbolic_eval__,
            "classification": __classification_symbolic_eval__,
        }

        self.generate_cex_constraint = models[model_type]
        self.symbolic_eval = symbolic_evaluation_models[model_type]

def generate_cex_constraint(seq, val, boolean, states, sigma_I, sigma_O, init_state, delta, output_fnc):
    """
    Given a counterexample constraint V(seq) != val, and a hypothesis h, generates the symbolic
    constraint induced by h.

    NOTE: We must explicitly include the initial state
    """
    q = init_state
    var_list = []
    var_list.append(q[0])
    for a in seq:
        q = delta[q][a]
        var_list.append(q[0])
    symbolic_sum = z3.Sum(var_list)
    return z3.Not(symbolic_sum == val) if not boolean else z3.Or(symbolic_sum == val)

def symbolic_eval(seq, states, sigma_I, sigma_O, init_state, delta, output_fnc):
    """
    Evaluates the symbolic and concrete hypothesis simultaneously to obtain the symbolic
    and concrete values of a sequence

    NOTE: We must explicitly include the initial state, even though it is 0. This is to take care
    of the empty sequence.
    """
    q = init_state
    total_sum = 0
    var_list = []
    total_sum += output_fnc[q]
    var_list.append(q[0])
    for a in seq:
        q = delta[q][a]
        var_list.append(q[0])
        total_sum += output_fnc[q]

    symbolic_sum = z3.Sum(var_list)
    return symbolic_sum, total_sum

def convert_to_equivalence_classes(eq_constraints, EC, RP, print_fnc=print):
    """
    NOTE: This function takes the set of eq_constraints and modifies
          the contents of EC and RP
    """
    print_fnc(f"[PRE EC]")
    for r in RP:
        print_fnc(EC[r])

    while len(eq_constraints) > 0:
        equality = eq_constraints.pop()
        print_fnc(f"   {equality}")
        ## Parse the equality
        L = equality.arg(0)
        R = equality.arg(1)

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
                        print_fnc(f"  -  Deleted {R_rep} (via eq)")
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
            print_fnc(f"  +  Added {rep} (via eq)")
    
    print_fnc(f"[POST EC]")
    for r in RP:
        print_fnc(EC[r])
    print_fnc(f"==[POST EC]==")

        ## NOTE: Now we can do things like check if "EC[a] is EC[b]", which tells us whether
        ## "a" and "b" belong to the same equivalence class or not.
        ## We can also perform substitution by using EC[a].repr() -- this will look up
        ## the representative for the equivalence class that "a" belongs to, and return that
        ## representative for us to use
