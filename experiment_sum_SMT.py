from collections import deque
import fractions
import itertools
import argparse
import random
import numpy as np
import networkx as nx
from datastructures import BijectiveIndexMapping, TableList, EquivalenceClass
from lstar_sum_SMT import SymbolicObservationTable
from lstar_sum_SMT import symbolic_lstar, concat
from lstar_utils import get_vars, range_prefixes
from Mealy import Mealy
from Moore import Moore
import z3
import exrex
import sympy
import os

class MooreMachineTeacher:
    """
    This defines a Moore machine teacher, and specifies the valuation model that the teacher uses
    """
    def __init__(self, moore_machine, seq_sample_size, init_state=None, forced_test=None, use_strong_feedback=False, valuation_model=None, gamma=None):
        self.moore_machine = moore_machine
        self.q0 = self.moore_machine.initial_state
        ## The input alphabet should be the power set of the input alphabet
        self.sigma_I = tuple(self.moore_machine.input_alphabet)
        ## The output alphabet should be the possible reward values that can be returned from the reward machine
        self.sigma_O = tuple(self.moore_machine.output_alphabet)
        self.rng = np.random.default_rng()
        self.seq_sample_size = seq_sample_size
        self.forced_seq = None
        if forced_test is not None:
            self.forced_seq = forced_test
        self.use_strong_feedback=use_strong_feedback

        self.models = {
            "sum": lambda a,b,c: a+b,
            "discountsum": lambda a,b,c: a+b*c,
            "prod": lambda a,b,c: a*b,
            "classification": lambda a,b,c: b,
        }

        if valuation_model is None or valuation_model not in self.models:
            raise ValueError("The teacher requires a valuation model")
        if valuation_model == "discountsum" and gamma is None:
            raise ValueError("Discount summation requires a discount factor")

        self.gamma = fractions.Fraction("1")
        if valuation_model != "discountsum":
            self.gamma = fractions.Fraction("1")
        else:
            if not isinstance(gamma, fractions.Fraction):
                raise ValueError("Need to ensure that gamma has a perfect rational representation")
            self.gamma = gamma

        self.valuation_model = self.models[valuation_model]
    
    def sample_sequences(self, quantity):
        """
        We can do random sequences, or we can sample sequences from the reward machine itself
        Lets do random sequences first
        """
        p = 0.2
        if self.forced_seq is not None:
            quantity = quantity - len(self.forced_seq) - 1
        else:
            quantity = quantity - 1

        seq_lengths = self.rng.geometric(p, quantity)
        sequences = []
        sequences.append(tuple())
        if self.forced_seq is not None:
            for seq in self.forced_seq:
                sequences.append(seq)
        for length in seq_lengths:
            ## Generate a random sequence of the desired length
            sequences.append(tuple(random.choices(self.sigma_I, k=length)))
        return sequences

    def evaluate_sequence(self, s):
        q = self.q0
        total = fractions.Fraction(self.moore_machine.output_table[q])
        factor = fractions.Fraction("1")
        for a in s:
            q = self.moore_machine.transitions[q][a]
            r = fractions.Fraction(self.moore_machine.output_table[q])
            factor = factor*self.gamma
            total = self.valuation_model(total, r, factor)
            #total += r

        return total

    def preference_query(self, s1, s2):
        r1 = self.evaluate_sequence(s1)
        r2 = self.evaluate_sequence(s2)
        if r1 == r2:
            return 0
        elif r1 > r2:
            return 1
        else:
            return -1

    def __equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Computes an empirical equivalence query (sampling-based)

        Observed Cases of equiv query:
            Init State = 3; Test Seq: (0, -2, 2)
        """
        def evaluate_hypothesis(seq):
            q = init_state
            total = output_fnc[q]
            for a in seq:
                q = delta[q][a]
                v = output_fnc[q]
                total += v
            return total

        sequences = self.sample_sequences(self.seq_sample_size)

        print("=== >>> Teacher Equiv Query <<< ===\n")
        print(f"  Teacher Init State: {self.q0}\n")
        print(f"  Testing Hypothesis:\n  {delta}\n  {output_fnc}\n")
        n_tested = 0
        for seq in sequences:
            teacher_output = self.evaluate_sequence(seq)
            learner_output = evaluate_hypothesis(seq)
            #print(f"  SEQ LEN: {len(seq)}  --  SEQ: {seq}\n   > TEACHER OUT: {teacher_output}\n   > LEARNER OUT: {learner_output}")
            n_tested += 1
            if teacher_output != learner_output:
                print(f"  After {n_tested} sequences, found a counter example:")
                print(f"  SEQ LEN: {len(seq)}  --  SEQ: {seq}\n   > TEACHER OUT: {teacher_output}\n   > LEARNER OUT: {learner_output}")
                return False, (seq, teacher_output)

        print(f"Tested {n_tested} sequences, passes Equivalence Check")
        return True, None
    
    def equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Performs a symbolic evaluation of whether two Moore machines are equivalent.

        This implements an optimized version of the Hopcroft-Karp algorithm. https://arxiv.org/abs/0907.5058

        Also here, have adopted the equivalence check from https://github.com/caleb531/automata/blob/main/automata/fa/dfa.py

        The input values in the function refer to the automata to be tested. The ground truth is the teacher's automata.
        """
        print(states)
        print(sigma_I)
        print(sigma_O)
        print(init_state)
        print(delta)
        print(output_fnc)

        if sigma_I != self.sigma_I:
            return ValueError("The teacher and learner have different input alphabets")

        initial_state_teacher = self.q0
        initial_state_learner = init_state

        ## In all operations below, we assume that the teacher is on the left side and the learner is on the right side.
        state_sets = nx.utils.union_find.UnionFind((initial_state_teacher, initial_state_learner))
        pair_stack = deque()

        state_sets.union( (1, initial_state_teacher, initial_state_learner), (0, initial_state_learner, initial_state_teacher))
        pair_stack.append((initial_state_teacher, initial_state_learner, tuple()))

        ## NOTE: This algorithm visits the states in both automata in a depth-first order. If the ordering of all the
        ## output values of the algorithm match, then both automata are equivalent. However, we will need to obtain a
        ## counterexample in the case that the automata are found inequivalent. This means that we will need to perform
        ## a DFS to the inequivalent state and record the sequence that gets us there. And return that as the cex.
        while pair_stack:
            q_T, q_L, cex = pair_stack.pop()
            print(f"Comparing {q_T} and {q_L}")

            ## NOTE: that HALT is an absorbing state in the teacher's Moore machine
            ## If the values are not the same, then the automata cannot be equivalent
            teacher_output = fractions.Fraction(self.moore_machine.output_table[q_T])
            if output_fnc[q_L] != teacher_output:
                if not self.use_strong_feedback:
                    print(f"Recieved Weak CEX: VALUE({cex}) != {teacher_output}")
                    return False, (cex, teacher_output, self.use_strong_feedback)
                else:
                    cex_val = self.evaluate_sequence(cex)
                    print(f"Recieved Strong CEX: VALUE({cex}) == {cex_val}")
                    return False, (cex, cex_val, self.use_strong_feedback)

            for letter in sigma_I:
                ## Here, r_T and r_L refer to set representatives
                n_T = self.moore_machine.transitions[q_T][letter]
                n_L = delta[q_L][letter]
                r_T = state_sets[(1, n_T, n_L)]
                r_L = state_sets[(0, n_L, n_T)]

                if r_T != r_L:
                    seq = list(cex)
                    seq.append(letter)
                    state_sets.union(r_T, r_L)
                    pair_stack.append((n_T, n_L, tuple(seq)))
        return True, None

def find_termination_states(mealy):
    ## Find all termination states
    termination_set = set()
    for state, transitions in mealy.transitions.items():
        if len(transitions) == 0:
            termination_set.add(state)
    
    ## Remove all termination states from the transition function
    for state in termination_set:
        del mealy.transitions[state]

    return tuple(termination_set)

def merge_termination_states(mealy):
    """
    This function finds all termination states, and merges them into a single state
    """
    ## Find all termination states
    termination_set = set()
    for state, transitions in mealy.transitions.items():
        if len(transitions) == 0:
            termination_set.add(state)

    ## Remove all termination states from the transition function
    for state in termination_set:
        del mealy.transitions[state]

    ## Select one termination state at the representative
    unique_termination_state = termination_set.pop()
    termination_set.add(unique_termination_state)

    ## Now we have only non-terminal states in the transition function
    for state, transitions in mealy.transitions.items():
        for formula, output in transitions.items():
            next_state, value = output
            ## Replace all terminal states with the terminal state representative
            if next_state in termination_set:
                transitions[formula] = tuple((unique_termination_state, value))

    ## Replace all terminal states in the state set with the unique terminal state
    new_states = set(mealy.states) - termination_set
    new_states.add(unique_termination_state)
    mealy.states = tuple(new_states)
    return unique_termination_state
 
def summarize_transitions(mealy, propositions):
    """
    Edges going to the same next state, and with the same output will be collapsed and summarized to the same edge.

    mealy.transitions: {state -> { letter: (state, value)}}
    """
    for source, transitions in mealy.transitions.items():
        ## Transition truth tables for this source state
        transition_truth_tables = dict()
        for letter, output in transitions.items():
            if output not in transition_truth_tables:
                transition_truth_tables[output] = set()
            transition_truth_tables[output].add(letter)
        ## source -> this output has a truth table.
        boolean_formula_transitions = dict()
        for output, truth_table in transition_truth_tables.items():
            ## Convert each truth table to a DNF
            dnf_terms = []
            for true_props in truth_table:
                unused_propositions = set(propositions)
                for prop in true_props:
                    unused_propositions.discard(prop)
                true_portion = sympy.And(*tuple(sympy.Symbol(prop) for prop in true_props))
                false_portion = sympy.And(*tuple((~sympy.Symbol(prop)) for prop in unused_propositions))
                conjunction = sympy.And(true_portion, false_portion)
                dnf_terms.append(conjunction)
            dnf = sympy.Or(*tuple(dnf_terms))
            dnf = sympy.simplify_logic(dnf, form="dnf")
            ## Stringify the DNF into the format that the reward machine likes to take in:
            ## Use ! for not, and make sure there is no white space in the expression
            boolean_formula_transitions[str(dnf).replace("~","!").replace(" ","")] = output
        mealy.transitions[source] = boolean_formula_transitions

def moore_machine_experiment(moore_machine, exp_file, save_file, samples, forced_test, strong_feedback, valuation_model, gamma, use_cc_obj, use_ve_obj, use_cex_expansion, use_ids):

    teacher = MooreMachineTeacher(moore_machine, samples, init_state=None,
                                  forced_test=forced_test, use_strong_feedback=strong_feedback, valuation_model=valuation_model, gamma=gamma)
    hypothesis, experimental_data = symbolic_lstar(teacher.sigma_I, teacher.sigma_O, teacher, valuation_model, gamma,
                                                    use_cc_obj=use_cc_obj,
                                                    use_ve_obj=use_ve_obj,
                                                    use_cex_expansion=use_cex_expansion,
                                                    use_ids=use_ids)
    num_pref_q, num_ineq, num_unique_seq, num_ECs, sat_time, sat_solves, max_sat_size, num_unique_table_vars, up_shape, lo_shape, num_equiv_q, cex_lengths, events = experimental_data

    exp_file.write(f"{len(hypothesis[0])}#{num_pref_q}#{num_equiv_q}#{num_ineq}#{num_ECs}#{sat_time}#{sat_solves}#{max_sat_size}#{num_unique_table_vars}#{num_unique_seq}#{up_shape}#{lo_shape}#{cex_lengths}#{tuple(events)}\n")

def do_sums_moore_machine_experiment(machines, args, test_name=None, trials=100, save=0):
    samples = 200
    prefix = f"lstar_exps/moore_machine_experiments/abstract-{samples}"
    is_strong = args.feedback_strength == "strong"
    if args.gamma is None:
        args.gamma = 1

    if test_name is not None:
        with open(f"abstract_{test_name}.csv.part-{save}-{save+1}", "a") as exp_f:
            exp_f.write("'Number of States'#'Num Pref Q'#'Num Equiv Q'#'Num Ineq'#'Num ECs'#'SAT Time'#'SAT Solves'#'MaxSAT Size'#'Num Unique Table Vars'#'Num Unique Sequences'#'Upper Dim'#'Lower Dim'#'CEX Lengths'#'Events'\n")
            for idx in range(0, trials):
                moore_machine_experiment(machines[test_name], exp_f, f"{prefix}/{test_name}.txt.{idx}", samples,
                forced_test=gen_forced_test_set(machines[test_name].input_alphabet, max_length=3),
                strong_feedback=is_strong, valuation_model=args.valuation_model, gamma=fractions.Fraction(f"{args.gamma}"),
                use_cc_obj=args.disable_cc_obj,
                use_ve_obj=args.disable_ve_obj,
                use_cex_expansion=args.enable_cex_expansion,
                use_ids=args.disable_ids)
    else:
        for k in machines:
            test_name = k
            with open(f"abstract_{test_name}.csv.part-{save}-{save+1}", "a") as exp_f:
                exp_f.write("'Number of States'#'Num Pref Q'#'Num Equiv Q'#'Num Ineq'#'Num ECs'#'SAT Time'#'SAT Solves'#'MaxSAT Size'#'Num Unique Table Vars'#'Num Unique Sequences'#'Upper Dim'#'Lower Dim'#'CEX Lengths'#'Events'\n")
                for idx in range(0, trials):
                    moore_machine_experiment(machines[test_name], exp_f, f"{prefix}/{test_name}.txt.{idx}", samples,
                    forced_test=gen_forced_test_set(machines[test_name].input_alphabet, max_length=3),
                    strong_feedback=is_strong, valuation_model=args.valuation_model, gamma=fractions.Fraction(f"{args.gamma}"),
                    use_cc_obj=args.disable_cc_obj,
                    use_ve_obj=args.disable_ve_obj,
                    use_cex_expansion=args.enable_cex_expansion,
                    use_ids=args.disable_ids)
        


def gen_forced_test_set(alphabet, max_length=3):
    """
    Given an alphabet of size N, generate all possible strings of length 1 up to length M that
    can be composed from the alphabet, and ensure they are returned in a list
    with shortest strings first and longest strings last.

    [tuple(s) for s in ("a","b","aa", "ab", "bb", "ba", "aaa", "aab", "aba", "abb", "bba", "bbb", "baa", "bab")]
    """
    cur_length = 0

    aggregated_strings = list()
    temp_list = [tuple()]

    while cur_length < max_length:
        temp_list = [el + tuple(s) for el in temp_list for s in alphabet ]
        aggregated_strings.extend(temp_list)
        cur_length += 1

    return aggregated_strings

def moore_machine_library():
    d = {
        "test_zero_A": Moore(
            [0],
            ["a"],
            [0],
            {
                0: {
                    "a": 0,
                }
            },
            0,
            {
                0: 0,
            }
        ),
        "test_zero_B": Moore(
            [0],
            ["a","b"],
            [0],
            {
                0: {
                    "a": 0,
                    "b": 0,
                }
            },
            0,
            {
                0: 0,
            }
        ),
        "test_one_A": Moore(
            [0, 1],
            ["a"], ## Cycle between 0 and 1
            [0, 1],
            {
                0: {
                    "a": 1,
                },
                1: {
                    "a": 0,
                },
            },
            0,
            {
                0: 0,
                1: 1,
            }
        ),
        "test_one_A_extra": Moore(
            [0, 1],
            ["a"], ## Cycle between 0 and 1
            [0, 1, 3, 4],
            {
                0: {
                    "a": 1,
                },
                1: {
                    "a": 0,
                },
            },
            0,
            {
                0: 0,
                1: 1,
            }
        ),
        "test_one_B": Moore(
            [0, 1],
            ["a","b"],
            [0, 1],
            {
                0: {
                    "a": 1,
                    "b": 0,
                },
                1: {
                    "a": 0,
                    "b": 1,
                },
            },
            0,
            {
                0: 0,
                1: 1,
            }
        ),
        "test_two_C": Moore(
            [0, 1, 2],
            ["a", "b", "c"],
            [0, 1, 2],
            {
                0: {
                    "a": 1,
                    "b": 0,
                    "c": 2,
                },
                1: {
                    "a": 0,
                    "b": 1,
                    "c": 2,
                },
                2: {
                    "a": 1,
                    "b": 2,
                    "c": 0,
                },
            },
            0,
            {
                0: 0,
                1: 1,
                2: 2,
            }
        ),
        "testB": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 1,
                    "b": 2,
                },
                2: {
                    "a": 3,
                    "b": 3,
                },
                3: {
                    "a": 3,
                    "b": 3,
                },
            },
            0,
            {
                0: 0,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testB2": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 3,
                    "b": 2,
                },
                2: {
                    "a": 3,
                    "b": 3,
                },
                3: {
                    "a": 3,
                    "b": 3,
                },
            },
            0,
            {
                0: 0,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testB3": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 3,
                    "b": 2,
                },
                2: {
                    "a": 0,
                    "b": 3,
                },
                3: {
                    "a": 3,
                    "b": 3,
                },
            },
            0,
            {
                0: 0,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testB4": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 3,
                    "b": 2,
                },
                2: {
                    "a": 0,
                    "b": 3,
                },
                3: {
                    "a": 2,
                    "b": 1,
                },
            },
            0,
            {
                0: 0,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testB5": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 3,
                    "b": 2,
                },
                2: {
                    "a": 0,
                    "b": 3,
                },
                3: {
                    "a": 0,
                    "b": 3,
                },
            },
            0,
            {
                0: 0,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testB6": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 3,
                    "b": 2,
                },
                2: {
                    "a": 0,
                    "b": 3,
                },
                3: {
                    "a": 0,
                    "b": 1,
                },
            },
            0,
            {
                0: 0,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testC_linear": Moore(
            [0, 1, 2, 3],
            ["a", "b", "c"],
            [0, 1, 2, 3],
            {
                0: {
                    "a": 1,
                    "b": 2,
                    "c": 3,
                },
                1: {
                    "a": 2,
                    "b": 3,
                    "c": 0,
                },
                2: {
                    "a": 3,
                    "b": 0,
                    "c": 1,
                },
                3: {
                    "a": 0,
                    "b": 1,
                    "c": 2,
                },
            },
            0,
            {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
            }
        ),
        "simple_rm_A": Moore(
            [0, 1, 2, 3, 4],
            ["a", "b", "c"],
            [0, 1],
            {
                0: {
                    "a": 1,
                    "b": 0,
                    "c": 2,
                },
                1: {
                    "a": 1,
                    "b": 3,
                    "c": 1,
                },
                2: {
                    "a": 4,
                    "b": 4,
                    "c": 4,
                },
                3: {
                    "a": 4,
                    "b": 4,
                    "c": 4,
                },
                4: {
                    "a": 4,
                    "b": 4,
                    "c": 4,
                },
            },
            0,
            {
                0: 0,
                1: 0,
                2: 0,
                3: 1,
                4: 0,
            }
        ),
        "simple_rm_B1": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d"],
            [0, 1],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                },
            },
            1,
            {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
                7: 0,
            }
        ),
        "simple_rm_B2": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e"],
            [0, 1],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                },
            },
            1,
            {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
                7: 0,
            }
        ),
        "simple_rm_B3": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f"],
            [0, 1],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                },
            },
            1,
            {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
                7: 0,
            }
        ),
        "simple_rm_B4": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f","g"],
            [0, 1],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                    "g": 2,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                    "g": 2,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                    "g": 6,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                    "g": 6,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                },
            },
            1,
            {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
                7: 0,
            }
        ),
        "simple_rm_B5": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f","g","h"],
            [0, 1],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                    "g": 2,
                    "h": 3,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                    "g": 2,
                    "h": 2,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                    "g": 6,
                    "h": 4,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                    "g": 6,
                    "h": 4,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                },
            },
            1,
            {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
                7: 0,
            }
        ),
        "simple_rm_B6": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f","g","h","i"],
            [0, 1],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                    "g": 2,
                    "h": 3,
                    "i": 1,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                    "g": 2,
                    "h": 2,
                    "i": 4,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                    "g": 6,
                    "h": 4,
                    "i": 4,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                    "g": 6,
                    "h": 4,
                    "i": 4,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                },
            },
            1,
            {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
                7: 0,
            }
        ),
        "simple_rm_B7": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f","g","h","i","j"],
            [0, 1],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                    "g": 2,
                    "h": 3,
                    "i": 1,
                    "j": 2,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                    "g": 2,
                    "h": 2,
                    "i": 4,
                    "j": 6,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                    "g": 6,
                    "h": 4,
                    "i": 4,
                    "j": 6,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                    "g": 6,
                    "h": 4,
                    "i": 4,
                    "j": 6,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                    "j": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                    "j": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                    "j": 7,
                },
            },
            1,
            {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
                7: 0,
            }
        ),
        "simple_rm_B8": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f","g","h","i","j","k"],
            [0, 1],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                    "g": 2,
                    "h": 3,
                    "i": 1,
                    "j": 2,
                    "k": 6,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                    "g": 2,
                    "h": 2,
                    "i": 4,
                    "j": 6,
                    "k": 6,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                    "g": 6,
                    "h": 4,
                    "i": 4,
                    "j": 6,
                    "k": 3,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                    "g": 6,
                    "h": 4,
                    "i": 4,
                    "j": 6,
                    "k": 5,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                    "j": 7,
                    "k": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                    "j": 7,
                    "k": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                    "j": 7,
                    "k": 7,
                },
            },
            1,
            {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 1,
                6: 0,
                7: 0,
            }
        ),
        "simple_office_t4_A": Moore(
            [0, 1, 2, 3, 4, 5, 6, 7],
            ["a", "b", "c"],
            [0, 1],
            {
                0: {
                    "a": 0,
                    "b": 1,
                    "c": 5,
                },
                1: {
                    "a": 2,
                    "b": 5,
                    "c": 1,
                },
                2: {
                    "a": 5,
                    "b": 2,
                    "c": 3,
                },
                3: {
                    "a": 3,
                    "b": 4,
                    "c": 5,
                },
                4: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                },
                5: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                },
                6: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                },
            },
            0,
            {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 1,
                5: 0,
                6: 0,
            }
        ),
        "simple_office_t4_B": Moore(
            [0, 1, 2, 3, 4, 5, 6, 7],
            ["a", "b", "c", "d"],
            [0, 1],
            {
                0: {
                    "a": 0,
                    "b": 1,
                    "c": 5,
                    "d": 5,
                },
                1: {
                    "a": 2,
                    "b": 5,
                    "c": 1,
                    "d": 5,
                },
                2: {
                    "a": 5,
                    "b": 2,
                    "c": 3,
                    "d": 5,
                },
                3: {
                    "a": 3,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                },
                4: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                },
                5: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                },
                6: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                },
            },
            0,
            {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 1,
                5: 0,
                6: 0,
            }
        ),
        "simple_office_t4_C": Moore(
            [0, 1, 2, 3, 4, 5, 6, 7],
            ["a", "b", "c", "d", "e"],
            [0, 1],
            {
                0: {
                    "a": 0,
                    "b": 1,
                    "c": 5,
                    "d": 5,
                    "e": 0,
                },
                1: {
                    "a": 2,
                    "b": 5,
                    "c": 1,
                    "d": 5,
                    "e": 1,
                },
                2: {
                    "a": 5,
                    "b": 2,
                    "c": 3,
                    "d": 5,
                    "e": 2,
                },
                3: {
                    "a": 3,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 5,
                },
                4: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                    "e": 6,
                },
                5: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                    "e": 6,
                },
                6: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                    "e": 6,
                },
            },
            0,
            {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 1,
                5: 0,
                6: 0,
            }
        ),
        "simple_craft_t105_A": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c","d"],
            [0, 1],
            {
                1: {
                    "a": 1,
                    "b": 2,
                    "c": 3,
                    "d": 5,
                },
                2: {
                    "a": 2,
                    "b": 2,
                    "c": 4,
                    "d": 4,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 3,
                    "d": 3,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 4,
                    "d": 4,
                },
                5: {
                    "a": 4,
                    "b": 5,
                    "c": 5,
                    "d": 5,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                },
            },
            1,
            {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 1,
                7: 0,
            }
        ),
        "test_zero_A_prod": Moore(
            [0],
            ["a"],
            [1],
            {
                0: {
                    "a": 0,
                }
            },
            0,
            {
                0: 1,
            }
        ),
        "test_zero_B_prod": Moore(
            [0],
            ["a","b"],
            [1],
            {
                0: {
                    "a": 0,
                    "b": 0,
                }
            },
            0,
            {
                0: 1,
            }
        ),
        "test_one_A_prod": Moore(
            [0, 1],
            ["a"], ## Cycle between 0 and 1
            [0, 1],
            {
                0: {
                    "a": 1,
                },
                1: {
                    "a": 0,
                },
            },
            0,
            {
                0: 1,
                1: 1,
            }
        ),
        "test_one_AA_prod": Moore(
            [0, 1],
            ["a"], ## Cycle between 0 and 1
            [0, 1, 2],
            {
                0: {
                    "a": 1,
                },
                1: {
                    "a": 0,
                },
            },
            0,
            {
                0: 1,
                1: 2,
            }
        ),
        "test_one_A_extra_prod": Moore(
            [0, 1],
            ["a"], ## Cycle between 0 and 1
            [0, 1, 3, 4],
            {
                0: {
                    "a": 1,
                },
                1: {
                    "a": 0,
                },
            },
            0,
            {
                0: 1,
                1: 1,
            }
        ),
        "test_one_AA_extra_prod": Moore(
            [0, 1],
            ["a"], ## Cycle between 0 and 1
            [0, 1, 3, 4],
            {
                0: {
                    "a": 1,
                },
                1: {
                    "a": 0,
                },
            },
            0,
            {
                0: 1,
                1: 4,
            }
        ),
        "test_one_B_prod": Moore(
            [0, 1],
            ["a","b"],
            [0, 1],
            {
                0: {
                    "a": 1,
                    "b": 0,
                },
                1: {
                    "a": 0,
                    "b": 1,
                },
            },
            0,
            {
                0: 1,
                1: 1,
            }
        ),
        "test_one_BB_prod": Moore(
            [0, 1],
            ["a","b"],
            [0, 1, 2],
            {
                0: {
                    "a": 1,
                    "b": 0,
                },
                1: {
                    "a": 0,
                    "b": 1,
                },
            },
            0,
            {
                0: 1,
                1: 2,
            }
        ),
        "test_two_C_prod": Moore(
            [0, 1, 2],
            ["a", "b", "c"],
            [0, 1, 2],
            {
                0: {
                    "a": 1,
                    "b": 0,
                    "c": 2,
                },
                1: {
                    "a": 0,
                    "b": 1,
                    "c": 2,
                },
                2: {
                    "a": 1,
                    "b": 2,
                    "c": 0,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 2,
            }
        ),
        "test_two_CC_prod": Moore(
            [0, 1, 2],
            ["a", "b", "c"],
            [0, 1, 2, 4],
            {
                0: {
                    "a": 1,
                    "b": 0,
                    "c": 2,
                },
                1: {
                    "a": 0,
                    "b": 1,
                    "c": 2,
                },
                2: {
                    "a": 1,
                    "b": 2,
                    "c": 0,
                },
            },
            0,
            {
                0: 1,
                1: 4,
                2: 2,
            }
        ),
        "testB_prod": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [1, 2, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 1,
                    "b": 2,
                },
                2: {
                    "a": 3,
                    "b": 3,
                },
                3: {
                    "a": 3,
                    "b": 3,
                },
            },
            0,
            {
                0: 1,
                1: 2,
                2: 3,
                3: 5,
            }
        ),
        "testB2_prod": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 3,
                    "b": 2,
                },
                2: {
                    "a": 3,
                    "b": 3,
                },
                3: {
                    "a": 3,
                    "b": 3,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testB3_prod": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 3,
                    "b": 2,
                },
                2: {
                    "a": 0,
                    "b": 3,
                },
                3: {
                    "a": 3,
                    "b": 3,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testB4_prod": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 3,
                    "b": 2,
                },
                2: {
                    "a": 0,
                    "b": 3,
                },
                3: {
                    "a": 2,
                    "b": 1,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testB5_prod": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 3,
                    "b": 2,
                },
                2: {
                    "a": 0,
                    "b": 3,
                },
                3: {
                    "a": 0,
                    "b": 3,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testB6_prod": Moore(
            [0, 1, 2, 3],
            ["a", "b"],
            [0, 1, 3, 5],
            {
                0: {
                    "a": 1,
                    "b": 2,
                },
                1: {
                    "a": 3,
                    "b": 2,
                },
                2: {
                    "a": 0,
                    "b": 3,
                },
                3: {
                    "a": 0,
                    "b": 1,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 3,
                3: 5,
            }
        ),
        "testC_linear_prod": Moore(
            [0, 1, 2, 3],
            ["a", "b", "c"],
            [0, 1, 2, 3],
            {
                0: {
                    "a": 1,
                    "b": 2,
                    "c": 3,
                },
                1: {
                    "a": 2,
                    "b": 3,
                    "c": 0,
                },
                2: {
                    "a": 3,
                    "b": 0,
                    "c": 1,
                },
                3: {
                    "a": 0,
                    "b": 1,
                    "c": 2,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 2,
                3: 3,
            }
        ),
        "simple_rm_A_prod": Moore(
            [0, 1, 2, 3, 4],
            ["a", "b", "c"],
            [0, 1, 7],
            {
                0: {
                    "a": 1,
                    "b": 0,
                    "c": 2,
                },
                1: {
                    "a": 1,
                    "b": 3,
                    "c": 1,
                },
                2: {
                    "a": 4,
                    "b": 4,
                    "c": 4,
                },
                3: {
                    "a": 4,
                    "b": 4,
                    "c": 4,
                },
                4: {
                    "a": 4,
                    "b": 4,
                    "c": 4,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 1,
                3: 7,
                4: 1,
            }
        ),
        "simple_rm_B1_prod": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d"],
            [0, 1, 3, 9],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                },
            },
            1,
            {
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 9,
                6: 1,
                7: 1,
            }
        ),
        "simple_rm_B2_prod": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e"],
            [0, 1, 3, 9],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                },
            },
            1,
            {
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 9,
                6: 1,
                7: 1,
            }
        ),
        "simple_rm_B3_prod": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f"],
            [0, 1, 3, 9],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                },
            },
            1,
            {
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 9,
                6: 1,
                7: 1,
            }
        ),
        "simple_rm_B4_prod": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f","g"],
            [0, 1, 3, 9],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                    "g": 2,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                    "g": 2,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                    "g": 6,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                    "g": 6,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                },
            },
            1,
            {
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 9,
                6: 1,
                7: 1,
            }
        ),
        "simple_rm_B5_prod": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f","g","h"],
            [0, 1, 3, 9],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                    "g": 2,
                    "h": 3,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                    "g": 2,
                    "h": 2,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                    "g": 6,
                    "h": 4,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                    "g": 6,
                    "h": 4,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                },
            },
            1,
            {
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 9,
                6: 1,
                7: 1,
            }
        ),
        "simple_rm_B6_prod": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f","g","h","i"],
            [0, 1, 3, 9],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                    "g": 2,
                    "h": 3,
                    "i": 1,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                    "g": 2,
                    "h": 2,
                    "i": 4,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                    "g": 6,
                    "h": 4,
                    "i": 4,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                    "g": 6,
                    "h": 4,
                    "i": 4,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                },
            },
            1,
            {
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 9,
                6: 1,
                7: 1,
            }
        ),
        "simple_rm_B7_prod": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c", "d", "e","f","g","h","i","j"],
            [0, 1, 3, 9],
            {
                1: {
                    "a": 1,
                    "b": 6,
                    "c": 2,
                    "d": 3,
                    "e": 6,
                    "f": 1,
                    "g": 2,
                    "h": 3,
                    "i": 1,
                    "j": 2,
                },
                2: {
                    "a": 6,
                    "b": 4,
                    "c": 2,
                    "d": 4,
                    "e": 6,
                    "f": 6,
                    "g": 2,
                    "h": 2,
                    "i": 4,
                    "j": 6,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 6,
                    "d": 3,
                    "e": 3,
                    "f": 3,
                    "g": 6,
                    "h": 4,
                    "i": 4,
                    "j": 6,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 4,
                    "f": 5,
                    "g": 6,
                    "h": 4,
                    "i": 4,
                    "j": 6,
                },
                5: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                    "j": 7,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                    "j": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                    "e": 7,
                    "f": 7,
                    "g": 7,
                    "h": 7,
                    "i": 7,
                    "j": 7,
                },
            },
            1,
            {
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 9,
                6: 1,
                7: 1,
            }
        ),
        "simple_office_t4_A_prod": Moore(
            [0, 1, 2, 3, 4, 5, 6, 7],
            ["a", "b", "c"],
            [0, 1, 8],
            {
                0: {
                    "a": 0,
                    "b": 1,
                    "c": 5,
                },
                1: {
                    "a": 2,
                    "b": 5,
                    "c": 1,
                },
                2: {
                    "a": 5,
                    "b": 2,
                    "c": 3,
                },
                3: {
                    "a": 3,
                    "b": 4,
                    "c": 5,
                },
                4: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                },
                5: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                },
                6: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 1,
                3: 1,
                4: 8,
                5: 1,
                6: 1,
            }
        ),
        "simple_office_t4_B_prod": Moore(
            [0, 1, 2, 3, 4, 5, 6, 7],
            ["a", "b", "c", "d"],
            [0, 1, 2],
            {
                0: {
                    "a": 0,
                    "b": 1,
                    "c": 5,
                    "d": 5,
                },
                1: {
                    "a": 2,
                    "b": 5,
                    "c": 1,
                    "d": 5,
                },
                2: {
                    "a": 5,
                    "b": 2,
                    "c": 3,
                    "d": 5,
                },
                3: {
                    "a": 3,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                },
                4: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                },
                5: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                },
                6: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 1,
                3: 1,
                4: 2,
                5: 1,
                6: 1,
            }
        ),
        "simple_office_t4_C_prod": Moore(
            [0, 1, 2, 3, 4, 5, 6, 7],
            ["a", "b", "c", "d", "e"],
            [0, 1, 3],
            {
                0: {
                    "a": 0,
                    "b": 1,
                    "c": 5,
                    "d": 5,
                    "e": 0,
                },
                1: {
                    "a": 2,
                    "b": 5,
                    "c": 1,
                    "d": 5,
                    "e": 1,
                },
                2: {
                    "a": 5,
                    "b": 2,
                    "c": 3,
                    "d": 5,
                    "e": 2,
                },
                3: {
                    "a": 3,
                    "b": 4,
                    "c": 5,
                    "d": 5,
                    "e": 5,
                },
                4: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                    "e": 6,
                },
                5: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                    "e": 6,
                },
                6: {
                    "a": 6,
                    "b": 6,
                    "c": 6,
                    "d": 6,
                    "e": 6,
                },
            },
            0,
            {
                0: 1,
                1: 1,
                2: 1,
                3: 1,
                4: 3,
                5: 1,
                6: 1,
            }
        ),
        "simple_craft_t105_A_prod": Moore(
            [1,2,3,4,5,6,7],
            ["a", "b", "c","d"],
            [1, 3],
            {
                1: {
                    "a": 1,
                    "b": 2,
                    "c": 3,
                    "d": 5,
                },
                2: {
                    "a": 2,
                    "b": 2,
                    "c": 4,
                    "d": 4,
                },
                3: {
                    "a": 4,
                    "b": 3,
                    "c": 3,
                    "d": 3,
                },
                4: {
                    "a": 6,
                    "b": 4,
                    "c": 4,
                    "d": 4,
                },
                5: {
                    "a": 4,
                    "b": 5,
                    "c": 5,
                    "d": 5,
                },
                6: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                },
                7: {
                    "a": 7,
                    "b": 7,
                    "c": 7,
                    "d": 7,
                },
            },
            1,
            {
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
                6: 3,
                7: 1,
            }
        ),
    }
    return d

def parse_args():
    parser = argparse.ArgumentParser(description='Provide Filenames')
    parser.add_argument('--trial', type=int, default=None,
                        help='Trial number')
    parser.add_argument('--trial_min', type=int,
                        help='Minimum trial unit')
    parser.add_argument('--trial_max', type=int,
                        help='Maximum trial unit')
    parser.add_argument('--domain', type=str,
                        help='Domain for learning')
    parser.add_argument('--task', type=str,
                        help='Task for learning')
    parser.add_argument('--test-name', type=str,
                        help='Moore Machine Library Key')
    exp_types = ["rm","moore"]
    feedback_strength = ["weak", "strong"]
    valuation_model = ["sum", "discountsum", "prod", "classification"]
    parser.add_argument('--experiment-type', type=str, choices=exp_types,
                        help='Experiment type (rm, moore)')
    parser.add_argument('--num-trials', type=int,
                        help='Number of trials to run')
    parser.add_argument('--feedback-strength', type=str, choices=feedback_strength,
                        help='Feedback strength (strong, weak)')
    parser.add_argument('--valuation-model', type=str, choices=valuation_model,
                        help='Valuation model (sum, discountsum, prod, classification)')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Discount factor to use with the discountsum valuation model')
    parser.add_argument('--disable-cc-obj', action='store_false',
                        help='Disables the closed and consistent objective')
    parser.add_argument('--disable-ve-obj', action='store_false',
                        help='Disables the variable equivalence objective')
    parser.add_argument('--enable-cex-expansion', action='store_true',
                        help='Enables expanding the table by counterexample')
    parser.add_argument('--stateless-execution', action='store_true', ## TODO
                        help='Enables stateless execution by not enforcing equivalence classes')
    parser.add_argument('--disable-ids', action='store_false',
                        help='Disables the iterative deepening search')
    args = parser.parse_args()
    if args.experiment_type not in exp_types:
        raise TypeError(f"Experiment Type (--experiment-type) must be specified, must be one of {exp_types}")
    if args.feedback_strength not in feedback_strength:
        raise TypeError(f"Feedback strength (--feedback-strength) must be specified, must be one of {feedback_strength}")
    if args.valuation_model not in valuation_model:
        raise TypeError(f"Valuation model (--valuation-model)  must be specified, must be one of {valuation_model}")
    if args.valuation_model == "discountsum" and args.gamma is None:
        raise TypeError(f"If valuation is discountsum, then discount factor (--gamma) must be specified")
    if not args.disable_cc_obj and not args.disable_ve_obj:
        raise TypeError(f"Both objectives cannot be disabled at the same time")

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.experiment_type == "moore":
        machines = moore_machine_library()
        do_sums_moore_machine_experiment(machines, args, test_name=args.test_name, trials=args.num_trials, save=args.trial)
    else:
        _a = 6
