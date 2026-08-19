import z3
import itertools
import copy
from collections import deque
from datastructures import BijectiveIndexMapping, TableList, EquivalenceClass
from lstar_utils import get_vars, range_prefixes, generate_cex_constraint, symbolic_eval


class Checkpoint:
    """
    The Checkpoint contains saves all information known at the previous timepoint.

    In particular, it contains observation table and the constraints prior to any
    substitutions induced by assumptions.
    """
    def __init__(self, obstable):
        ## TODO: Tweak exactly what information should be restored in the checkpoint
        self.constraints = copy.deepcopy(obstable.constraints)
        self.upper = obstable.table_upper.clone() #copy.deepcopy(obstable.table_upper)
        self.lower = obstable.table_lower.clone() #copy.deepcopy(obstable.table_lower)
        self.prefix = copy.deepcopy(obstable.prefix_set)
        self.suffix = copy.deepcopy(obstable.suffix_set)
        self.prefix_alpha = copy.deepcopy(obstable.prefix_alpha_set)

    def restore(self, obstable):
        """
        NOTE: Non-unified ground truth constraints NEVER need to be restored. Only assumptions need to be undone.
        """
        #obstable.constraints = self.constraints
        obstable.table_upper = self.upper.clone()
        obstable.table_lower = self.lower.clone()
        obstable.prefix_set = self.prefix
        obstable.suffix_set = self.suffix
        obstable.prefix_alpha_set = self.prefix_alpha

    def __str__(self):
        s = "[CHECKPOINT]\n"
        s+= "     --- UPPER TABLE ---\n"
        s+= f"{self.upper}\n"
        s+= "     --- LOWER TABLE ---\n"
        s+= f"{self.lower}\n"
        s+= f"    --- RECORDED CONSTRAINTS ---\n"
        s+= self.__fmt_constraints__()
        s+= "[END CHECKPOINT]\n"
        return s

    def __fmt_constraints__(self):
        """
        This follows the constraint formatting of the SymbolicObservation table
        """
        EC = self.constraints["EC"]
        reps = set()
        for k, v in EC.items():
            reps.add(v.repr())

        s = f" --> NUMBER OF GROUND_TRUTH EQUIVALENCE CLASSES: {len(reps)}\n"
        counter = 1
        for v in reps:
            s += f"EC{counter}:\n{EC[v]}\n"
            counter+=1

        RP = self.constraints["repr"]
        s = f" --> NUMBER OF KNOWN GROUND_TRUTH REPR VALUES: {len(RP)}\n"
        s += f"  REPR VALUES:\n"
        for k, v in RP.items():
            s+= f"    {k}=={v}\n"
        
        GRD_INEQ = self.constraints["seq_ineq"]
        GRD_EQUL = self.constraints["seq_eq"]
        s = f" --> NUMBER OF GROUND_TRUTH CONSTRAINTS: {len(GRD_EQUL) + len(GRD_INEQ)}\n"
        s += f"  INEQUALITIES:\n"
        for v in GRD_INEQ:
            s += f"    {v}\n"
        s += f"  EQUALITIES:\n"
        for v in GRD_EQUL:
            s += f"    {v}\n"
        return s
        

class AtomicAssumptionStack:
    """
    The AtomicAssumptionStack contains 
        Checkpoint
        Stack of individual assumptions that are made during a symbolic fill
    """

    def __init__(self, obstable):
        """
        Creates a new atomic assumption stack, to be used only during a symbolic fill.
        """
        self.checkpoint = Checkpoint(obstable)
        self.assumption_stack = deque()

    def __iter__(self):
        for x in self.assumption_stack:
            yield x

    def is_empty(self):
        return True if len(self.assumption_stack) == 0 else False

    def all_false(self):
        """
        Checks if all assumptions are false in this stack
        """
        for b, _, _ in self.assumption_stack:
            if b:
                return False
        return True

    def push(self, assumption):
        """
        Assumption is as tuple : (Boolean, var1, var2)
        The semantics behind this are:
            Boolean = True or False
            If Boolean is True, then var1 == var2
            If Boolean is False, then var1 != var2
        """
        self.assumption_stack.append(assumption)

    def pop(self):
        return self.assumption_stack.pop()

    def __len__(self):
        return len(self.assumption_stack)

    def __str__(self):
        s = f"  [ATOMIC_ASSUMPTION_STACK]\n"
        s+= f"{self.checkpoint}"
        s+= f"  Number of Atomic Assumptions: {len(self)}\n"
        s+= "   [Stack Bottom]\n"
        for b, v1, v2 in self.assumption_stack:
            if b:
                s+= f"    {v1}=={v2}\n"
            else:
                s+= f"    {v1}!={v2}\n"
        s+= "   [Stack Top]\n"
        s+= "   [===ATOMIC_ASSUMPTION_STACK===]\n"
        return s

    def last(self):
        return self.assumption_stack[-1]

    def string_assumptions_only(self):
        s = f"  [ATOMIC_ASSUMPTION_STACK]\n"
        for b, v1, v2 in self.assumption_stack:
            if b:
                s+= f"    {v1}=={v2}\n"
            else:
                s+= f"    {v1}!={v2}\n"
        s+= "   [===ATOMIC_ASSUMPTION_STACK===]\n"
        return s

class LocalAssumptionStack:
    """
    The LocalAssumptionStack contains
        Stack of AtomicAssumptionStacks
        Corresponding Hypothesis

    A LocalAssumptionStack contains all assumptions that are made between a pair of hypotheses
    """
    def __init__(self):
        self.stack = deque()
        self.hypothesis = None

    def __len__(self):
        return len(self.stack)

    def last(self):
        return self.stack[-1]

    def first(self):
        return self.stack[0]

    def push(self, atomic_assumption_stack):
        self.stack.append(atomic_assumption_stack)

    def pop(self):
        return self.stack.pop()

    def set_hypothesis(self, hypothesis):
        """
        Saves the hypothesis for evaluation
        """
        self.hypothesis = hypothesis

    def backtrack(self):
        """
        Performs backtracking on the local assumption stack.
        The local assumption stack consists of several atomic assumption stacks.

        While it seems plausible to backtrack through the entire stack, what we can do instead is simply
        just jump to the earliest checkpoint in the local assumption stack, and start from there.

        We may be able to perform more sophisticated backtracking, based on unsat cores, but for now,
        let's just stick with going back to the safe known working checkpoint.
        """
        print(f"[LOCAL_ASSUMPTION_STACK]: Entered LocalAssumptionStack.backtrack()")
        print(f"  Length: {len(self)}")
        print(f"  Saved Hypothesis: {self.hypothesis}")
        print(f"  Top of Stack (most recent):\n   {self.last()}")
        print(f"  Bot of Stack (oldest)     :\n   {self.first()}")
        top_atomic_assumption_stack = self.first()
        self.stack = deque()
        self.hypothesis = None
        return top_atomic_assumption_stack
    
    def add_assumptions_to_solver(self, solver):
        solver.push()
        for atomic_assumption_stack in self.stack:
            for assumption in atomic_assumption_stack.assumption_stack:
                b, v1, v2 = assumption
                solver.add(v1 == v2) if b else solver.add(v1!=v2)

    def track_assumptions_in_solver(self, solver):
        assumption2name = dict()
        name2assumption = dict()
        for atomic_assumption_stack in self.stack:
            for assumption in atomic_assumption_stack.assumption_stack:
                b, v1, v2 = assumption
                if assumption not in assumption2name:
                    name = None
                    expr = None
                    if b:
                        expr = v1==v2
                    else:
                        expr = v1!=v2
                    name = str(expr)
                    assumption2name[assumption] = name
                    name2assumption[name] = expr
                    solver.assert_and_track(expr, name)
   
    def __iter__(self):
        for atomic_assumption_stack in self.stack:
            for assumption in atomic_assumption_stack:
                yield assumption
 
    def assumptions(self):
        """
        Returns all assumptions that have been made in this stack
        """
        for atomic_assumption_stack in self.stack:
            for assumption in atomic_assumption_stack.assumption_stack:
                b, v1, v2 = assumption
                yield v1 == v2 if b else v1 != v2
    
    def string_assumptions_only(self):
        s = f" [LOCAL ASSUMPTION STACK]"
        for atomic_assumption_stack in self.stack:
            s += atomic_assumption_stack.string_assumptions_only()
        s += " [==LOCAL ASSUMPTION STACK]"
        return s

class GlobalAssumptionStack:
    """
    The GlobalAssumptionStack contains
        Stack of LocalAssumptions

    The GlobalAssumptionStack contains all the assumptions that were made between initialization
    and the current state at this particular point in the algorithm's execution.
    """
    def __init__(self):
        self.stack = deque()

    def __len__(self):
        return len(self.stack)

    def push(self, local_stack):
        self.stack.append(local_stack)

    def pop(self):
        return self.stack.pop()

    def last(self):
        return self.stack[-1]

    def __iter__(self):
        """
        Returns all assumptions that have been made in this stack
        """
        for local_assumption_stack in self.stack:
            for atomic_assumption_stack in local_assumption_stack.stack:
                for assumption in atomic_assumption_stack:
                    yield assumption

    def track_assumptions_in_solver(self, solver):
        assumption2name = dict()
        name2assumption = dict()
        for local_assumption_stack in self.stack:
            for atomic_assumption_stack in local_assumption_stack.stack:
                for assumption in atomic_assumption_stack.assumption_stack:
                    b, v1, v2 = assumption
                    if assumption not in assumption2name:
                        name = None
                        expr = None
                        if b:
                            expr = v1==v2
                        else:
                            expr = v1!=v2
                        name = str(expr)
                        assumption2name[assumption] = name
                        name2assumption[name] = expr
                        solver.assert_and_track(expr, name)

    def add_assumptions_to_solver(self, solver):
        for local_assumption_stack in self.stack:
            solver.push()
            for atomic_assumption_stack in local_assumption_stack.stack:
                for assumption in atomic_assumption_stack.assumption_stack:
                    b, v1, v2 = assumption
                    solver.add(v1 == v2) if b else solver.add(v1!=v2)

    def assumptions(self):
        """
        Returns all assumptions that have been made in this stack
        """
        for local_assumption_stack in self.stack:
            for atomic_assumption_stack in local_assumption_stack.stack:
                for assumption in atomic_assumption_stack.assumption_stack:
                    b, v1, v2 = assumption
                    yield v1 == v2 if b else v1 != v2

    def string_assumptions_only(self):
        s = f"[GLOBAL ASSUMPTION STACK]"
        for local_assumption_stack in self.stack:
            s += local_assumption_stack.string_assumptions_only()
        s += f"[==GLOBAL ASSUMPTION STACK==]"
        return s

    def repr_hypothesis_stack(self):
        s = ""
        count = 0
        for las in self.stack:
            s += f"Stack Index: {count}\n  H: {las.hypothesis}\n"
        return s

    def backtrack(self, solver, constraints):
        """
        This backtrack function identifies the earliest LocalAsssumptionStack in which the assumptions cannot
        be satisfied.
        """
        print(f"[GLOBAL_ASSUMPTION_STACK]: Entered GlobalAssumptionStack.backtrack()")
        print(f"[GLOBAL_ASSUMPTION_STACK]: Length: {len(self)}")
        print(self.repr_hypothesis_stack())
        ## Check if the assumptions can be satisfied
        self.add_assumptions_to_solver(solver)

        ## Do big steps. We check if previous hypothesis. If it fails, we pop it. Once we encounter a non-failing hypothesis, we stop.
        bad_local_stack = None
        while len(self) > 0:
            solver.push()
            last_local_stack = self.last()
            ## NOTE: The generated concrete CEX constraint will be in terms of representatives because the hypothesis is in terms of representatives.
            for cex, concrete_val, boolean in constraints["seq_sym"]:
                induced_cex_constraint = generate_cex_constraint(cex, concrete_val, boolean, *(last_local_stack.hypothesis))
                solver.add(induced_cex_constraint) ## Add the induced constraints

            print(f"[GLOBAL_ASSUMPTION_STACK]: Testing hypothesis {last_local_stack.hypothesis}")
            print(f"[GLOBAL_ASSUMPTION_STACK]: Checking constraints\n{solver}")
            sat_unsat = solver.check()
            if sat_unsat == z3.unsat:
                print(f"[GLOBAL_ASSUMPTION_STACK]: UNSAT, popping.")
                ## This means this hypothesis does not work.
                bad_local_stack = self.pop() ## Pop bad hypothesis
                solver.pop() ## Pop induced constraint from hypothesis
                solver.pop() ## Pop local assumptions
            else:
                print(f"[GLOBAL_ASSUMPTION_STACK]:   SAT, the most recent hypothesis works")
                ## The hypothesis works. We don't need to pop anymore.
                ## We have backtracked to the appropriate LocalAssumptionStack where the a conflict was first
                ## identified, so we can restart our backtracking within the LocalAssumptionStack.
                return bad_local_stack
        return bad_local_stack
