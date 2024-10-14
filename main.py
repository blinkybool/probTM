'''
    Turing Machines.

    # Internal Representation
    Symbols, states, directions are integers

    0 is the blank symbol
    0 is the initial state
    1 is the halt state
    0 is stay (no move)
    1 is left
    2 is right
'''

from dataclasses import dataclass
from jaxtyping import Array, Int, Float, Bool, PRNGKeyArray
import jax
import jax.numpy as jnp
from typing import List
import numpy as np
import readchar
import shutil

@dataclass
class TuringMachineSpec:
    delta: Int[Array, "Q S 3"]
    tape_alphabet: List[str]
    states: List[str]

def make_TM_spec(
        transitions: List[tuple[str, str, str, str, str]],
        symbols: str,
        initial_state: str,
        halt_state: str,
        blank: str,
        left: str,
        right: str,
        stay: str,
    ) -> TuringMachineSpec:
    assert initial_state != halt_state, "Haven't thought about this case"
    assert blank not in symbols, "Don't add the blank symbol to the string of symbols"
    states = []
    inputs = set()

    converted = []
    for transition in transitions:
        if len(transition) == 5:
            converted.append(transition)
        elif len(transition) == 4:
            state, reads, action, next_state = transition
            assert all(read in symbols+blank for read in reads)
            if action in {stay, left, right}:
                for read in reads:
                    converted.append((state, read, read, action, next_state))
            else:
                assert len(action) == 2, f"Bad action {action}"
                assert action[0] == 'P', f"Bad action {action}"
                assert action[1] == blank or action[1] in symbols, f"Bad action {action}"
                for read in reads:
                    converted.append((state, read, action[1], stay, next_state))
        else:
            raise ValueError(f"Bad transition {transition}")

    for state, read, write, move, next_state in converted:
        assert read == blank or read in symbols, f"Input symbol {read} in {(read, state, write, next_state, move)} not found in provided symbols"
        assert write == blank or write in symbols, f"Output symbol {write} in {(read, state, write, next_state, move)} not found in provided symbols"
        assert move in {left, right, stay}, f"Bad move {move} in {(read, state, write, next_state, move)}"
        if (state,read) in inputs:
            raise ValueError(f"Duplicate input {(state, read)} in transitions")
        inputs.add((state,read))

        if state != initial_state and state != halt_state and state not in states:
            states.append(state)
        if next_state != initial_state and next_state != halt_state and next_state not in states:
            states.append(next_state)

    tape_alphabet = blank + symbols
    states = [initial_state, halt_state] + list(states)
    symbol_to_int = {s: i for i, s in enumerate(tape_alphabet)}
    state_to_int = {q: i for i, q in enumerate(states)}
    move_to_int = {stay: 0, left: 1, right: 2}

    delta = np.zeros((len(states), len(tape_alphabet), 3), dtype=jnp.uint8)
    for state, read, write, move, next_state in converted:
        delta[state_to_int[state], symbol_to_int[read]] = np.array([symbol_to_int[write], move_to_int[move], state_to_int[next_state]], dtype=jnp.uint8)

    # Set all unspecified transitions to halt and stay
    for i, q in enumerate(states):
        for j, s in enumerate(tape_alphabet):
            if (q,s) not in inputs:
                delta[i, j] = np.array([j, 0, 1], dtype=jnp.uint8)

    return TuringMachineSpec(delta, tape_alphabet, states)

def step(tape: Int[Array, "t"], head: int, state: int, delta: Int[Array, "S Q 3"]) -> tuple[Int[Array, "t"], int, int]:
    read = tape[head]
    write, move, new_state = delta[state, read]
    tape  = jnp.where(state == 1, tape,  tape.at[head].set(write))
    dir   = jnp.where(move ==  0, 0,     jnp.where(move == 1, -1, 1))
    head  = jnp.where(state == 1, head,  head + dir)
    state = jnp.where(state == 1, state, new_state)
    return (tape, head, state)

def multi_step(
        start_tape: Int[Array, "t"],
        start_head: int,
        start_state: int,
        delta: Int[Array, "Q S 3"],
        num_steps: int
    ) -> tuple[Int[Array, "h t"], Int[Array, "h"], Int[Array, "h"]]:

    def body(config, _):
        stepped = step(*config, delta)
        return (stepped, stepped)

    _, (tapes, heads, states) = jax.lax.scan(body, (start_tape, start_head, start_state), length=num_steps)
    return tapes, heads, states

def run_TM(spec: TuringMachineSpec, input_string: str, max_steps: int = 1000) -> tuple[str, str]:
    delta = jnp.array(spec.delta)

    head_start = max(max_steps, len(input_string))
    init_tape = jnp.zeros(2*head_start+1, dtype=jnp.uint8)
    init_tape = init_tape.at[head_start:head_start+len(input_string)].set(jnp.array([spec.tape_alphabet.index(s) for s in input_string], dtype=jnp.uint8))

    tapes, heads, states = jax.jit(multi_step, static_argnames='num_steps')(init_tape, head_start, jnp.astype(0, jnp.uint8), delta, max_steps)
    # Prepend initial values
    tapes = jnp.concatenate([jnp.expand_dims(init_tape, 0), tapes])
    heads = jnp.concatenate([jnp.array([head_start], dtype=jnp.uint32), heads])
    states = jnp.concatenate([jnp.array([0], dtype=jnp.uint8), states])
    return tapes, heads, states

def print_execution(spec: TuringMachineSpec, tapes: Int[Array, "h t"], heads: Int[Array, "h"], states: Int[Array, "h"]) -> str:
    first = jnp.min(heads)
    last = jnp.max(heads)
    
    longest_state_name = max(len(q) for q in spec.states)
    for i, (tape, head, state) in enumerate(zip(tapes, heads, states)):
        tape_chars = [spec.tape_alphabet[i] for i in tape[first:head]]
        tape_chars.append(f"\033[48;5;240m{spec.tape_alphabet[tape[head]]}\033[0m")
        tape_chars.extend([spec.tape_alphabet[i] for i in tape[head+1:last]])
        print(f"{i:<4}[{spec.states[state]:<{longest_state_name}}] " + "".join(tape_chars))
        if state == 1:
            break

def all_deltas(num_states: int, num_symbols: int) -> Int[Array, "b Q S 3"]:
    '''
    Generate all possible delta functions.
    A delta function is a 3D array with a state axis, then symbol axis, and the inner
    values are length 3 arrays consisting of a symbol, state, and a 1 or -1 as the third value.
    We want to output an array of all such deltas.
    '''

    # Generate all possible states
    states = jnp.arange(num_states)
    symbols = jnp.arange(num_symbols)
    moves = jnp.array([0, 1, 2])
    # Generate all possible outputs
    outputs = jnp.stack(jnp.meshgrid(symbols, moves, states, indexing='ij'), axis=-1)
    # Partially flatten to array of output triples
    outputs = outputs.reshape(-1, 3)
    num_input_states = num_states
    num_input_symbols = len(symbols)

    # Generate indices for all possible combinations
    indices = jnp.arange(len(outputs))
    
    # Create a meshgrid of indices for each position in the delta function
    grid = jnp.stack(jnp.meshgrid(*[indices] * (num_input_states * num_input_symbols)), axis=-1)
    
    # Reshape the grid to match the desired output shape
    grid = grid.reshape(-1, num_input_states, num_input_symbols)
    
    # Use the indices to select the corresponding output triples
    deltas = outputs[grid]
    
    return deltas

def encode(spec: TuringMachineSpec, blank: str) -> str:
    delta = spec.delta

    expressions = []
    for state in range(delta.shape[0]):
        for read in range(delta.shape[1]):
            if state == 1:
                continue

            write, move, next_state = delta[state, read]
            
            if next_state == 1 and move == 0 and write == read:
                continue

            # Re-index so 1 is the initial state and the halt state == number of states
            # since that has no outgoing transitions
            state_index = {0: 1, 1: delta.shape[0]}.get(int(state), int(state))
            next_state_index = {0: 1, 1: delta.shape[0]}.get(int(next_state), int(next_state))
            
            state_enc = 'D' + 'A' * state_index
            next_state_enc = 'D' + 'A' * next_state_index
            read_enc = 'D' + 'C' * read
            write_enc = 'D' + 'C' * write
            move_enc = 'NLR'[move]
            expressions.append(state_enc + read_enc + write_enc + move_enc + next_state_enc)
    
    description = ";" + ";".join(expressions)

    return "əə" + blank.join(description + "$")

def make_UTM():
    symbols = 'ACDNLR01uvwxyzə;:$'
    blank = '_'
    notblank = symbols
    wild = "_" + symbols

    def exclude(*args):
        return ''.join(s for s in wild if s not in ''.join(args))
    
    def branch(state, *branches):
        all_reads = set()
        for reads, actions, final_state in branches:
            for read in reads:
                assert read not in all_reads, f"Read symbol {read} appears in more than one branch"
                all_reads.add(read)

        for i, (reads, actions, final_state) in enumerate(branches):
            if len(actions) == 0:
                yield (state, reads, 'N', final_state)
                continue
            if len(actions) == 1 and actions[0] in "LRN":
                yield (state, reads, actions[0], final_state)
                continue

            sub_step_reads = reads

            for j, action in enumerate(actions):
                if j == 0:
                    current_state = state
                else:
                    current_state = state + '?' + str(i+1) + ':' + str(j)
                if j == len(actions) - 1:
                    next_state = final_state
                else:
                    next_state = state + '?' + str(i+1) + ':' + str(j+1)

                yield (current_state, sub_step_reads, action, next_state)
                
                sub_step_reads = wild

    def f(C,B,a):
        q = f'f({C},{B},{a})'
        q1 = f'f1({C},{B},{a})'
        q2 = f'f2({C},{B},{a})'

        yield from branch(q,
            ('ə', ('L'), q1),
            (exclude('ə'), ('L',), q),
        )
        yield from branch(q1,
            (a, (), C),
            (exclude(a,blank), ('R',), q1),
            (blank, ('R',), q2),
        )
        yield from branch(q2,
            (a, (), C),
            (exclude(a,blank), ('R',), q1),
            (blank, ('R',), B),
        )

    def e(*args):
        if len(args) == 3:
            C, B, a = args
            yield (f'e({C},{B},{a})', wild, 'N', f'f(e1({C},{B},{a}),{B},{a})')
            yield from f(f'e1({C},{B},{a})',B,a)
            yield (f'e1({C},{B},{a})', a, blank, 'N', C)
        else:
            B, a = args
            yield (f'e({B},{a})', wild, 'N', f'e(e({B},{a}),{B},{a})')
            yield from e(f'e({B},{a})', B, a)

    def g(C,a):
        '''
        Find the last complete configuration by going right until a blank
        is found on an F-square, then backtrack left until 'D', 'C' or 'A' is
        found, then keep going left and stop on ':'.

        I don't know whether there might be a simulated output here to skip
        over, like ":0" at the end, but we assume it's possible for safety

        Assumes we are starting on an F-square
        '''

        q = f'g({C},{a})'
        q1 = f'g1({C},{a})'
        q2 = f'g2({C},{a})'

        yield from branch(q,
            (notblank, ('R','R'), q),
            (blank, (), q1)
        )

        yield from branch(q1, 
            (exclude("DCA"), ('L', 'L'), q1),
            ("DCA", ('L', 'L'), q2)
        )

        yield from branch(q2,
            (exclude(':'), ('L', 'L'), q2),
            (':', (), C)
        )


    def con(C,a):
        q = f'con({C},{a})'
        q1 = f'con1({C},{a})'
        q2 = f'con2({C},{a})'
        Pa = 'P' + a
        
        yield from branch(q,
            (exclude('A'), ('R', 'R'), q),
            ('A', ('L', Pa, 'R'), q1)
        )

        yield from branch(q1,
            ('A', ('R', Pa, 'R'), q1),
            ('D', ('R', Pa, 'R'), q2),
            ('_', ('R', 'R'), C), # A guess at a fix
            # (exclude('DCA'), ('R', 'R'), C) # Not in turing's specification
        )

        yield from branch(q2,
            ('C', ('R', Pa, 'R'), q2),
            (exclude('C'), ('R', 'R'), C)
        )
    
    def l(C):
        yield (f'l({C})', wild, 'L', C)

    def r(C):
        yield (f'r({C})', wild, 'R', C)

    def fl(C,B,a):
        yield (f'fl({C},{B},{a})', wild, 'N', f'f(l({C}),{B},{a})')
        yield from f(f'l({C})',B,a)
        yield from l(C)

    def fr(C,B,a):
        yield (f'fr({C},{B},{a})', wild, 'N', f'f(r({C}),{B},{a})')
        yield from f(f'r({C})',B,a)
        yield from r(C)

    def cp(C,A,E,a,b):
        q = f'cp({C},{A},{E},{a},{b})'
        q1 = f'cp1({C},{A},{b})'
        q2 = lambda g: f'cp2({C},{A},{g},{A},{b})'

        yield (q, wild, 'N', f'fl({q1},f({A},{E},{b}),{a})')
        yield from fl(q1,f'f({A},{E},{b})',a)        
        yield from f(A,E,b)
        for g in notblank:
            yield (q1, g, 'N', f'fl({q2(g)},{A},{b})')
            yield from fl(q2(g),A,b)
            yield from branch(q2(g),
                (g, (), C),
                (exclude(g), (), A)
            )


    def cpe(*args):
        if len(args) == 5:
            C,A,E,a,b = args
            yield (f'cpe({C},{A},{E},{a},{b})', wild, 'N', f'cp(e(e({C},{C},{b}),{C},{a}),{A},{E},{a},{b})')
            yield from cp(f'e(e({C},{C},{b}),{C},{a})',A,E,a,b)
            yield from e(f'e({C},{C},{b})',C,a)
            yield from e(C,C,b)
        else:
            A,E,a,b = args
            yield (f'cpe({A},{E},{a},{b})', wild, 'N', f'cpe(cpe({A},{E},{a},{b}),{A},{E},{a},{b})')
            yield from cpe(f'cpe({A},{E},{a},{b})',A,E,a,b)

    transitions = [
        ('b', wild, 'N', 'f(b1,b1,$)'),
        *f('b1','b1','$'),
        *branch('b1', 
            (wild, ('R','R','P:','R','R','PD','R','R','PA'), 'anf')
        ),
        ('anf', wild, 'N', 'g(anf1,:)'),
        *g('anf1',':'),
        ('anf1', wild, 'N', 'con(kom,y)'),
        *con('kom','y'),
        *branch('kom',
            (";", ('R', 'Pz', 'L'), 'con(kmp,x)'),
            ("z", ('L', 'L'), 'kom'),
            (exclude(";z"), ('L'), 'kom'),
        ),
        *con('kmp','x'),
        ('kmp', wild, 'N', 'cpe(e(e(kom,y),x),sim,x,y)'),
        *cpe('e(e(kom,y),x)','sim','x','y'),
        *e('e(kom,y)','x'),
        *e('kom','y'),
    ]

    return make_TM_spec(
        symbols=symbols,
        transitions=transitions,
        initial_state='b',
        halt_state='halt',
        blank='_',
        left='L',
        right='R',
        stay='N'
    )


def interactive_TM(spec, tapes, heads, states, lower_index=None):
    first = jnp.min(heads)
    if lower_index is not None:
        first = heads[0]+lower_index
    last = jnp.max(heads)
    halt_step = jnp.argmax(states == 1)
    if states[halt_step] == 1:
        last_step = halt_step
    else:
        last_step = len(tapes)-1

    tape_length = last+1 - first

    tape_changed_indices = jnp.where(jnp.any(tapes[1:] != tapes[:-1], axis=1))[0] + 1

    tape_symbols = np.array(list(spec.tape_alphabet))[tapes].astype('U32')
    tape_strs = []
    for tape, head in zip(tape_symbols, heads):

        tape_chars = list(tape)
        for i in range(first, last+1):
            if i % 2 != 0 and tape_chars[i] != '_':
                # Surrounded in ANSI escape code for green text
                tape_chars[i] = f"\033[38;5;34m{tape_chars[i]}\033[0m"
                

        if head < first:
            tape_strs.append(''.join(tape_chars[first:last+1]))
        else:
            tape_strs.append(
                ''.join(tape_chars[first:head])
                + f"\033[48;5;240m{tape_chars[head]}\033[0m"
                + ''.join(tape_chars[head+1:last+1])
            )

    step = 0
    while True:
        tape_str = tape_strs[step]
        state = states[step]

        cols, _ = shutil.get_terminal_size()

        lines = [
            f"Step: {step}/{last_step} | Press Q to quit",
            f"State: {spec.states[state]}",
            tape_str
        ]
        for line in lines:
            print('\33[2K\r' + line)
        
        lengths = [len(line) for line in lines]
        lengths[-1] = tape_length

        key = readchar.readkey()

        if key.lower() == 'q':
            break
        elif key == readchar.key.UP:
            if step > 0:
                step -= 1
        elif key == readchar.key.DOWN:
            if step < last_step:
                step += 1
        elif key == readchar.key.LEFT:
            if step > 0:
                prev_changes = tape_changed_indices[tape_changed_indices < step] - 1
                if len(prev_changes) > 0:
                    step = prev_changes[-1]
                else:
                    step = 0
        elif key == readchar.key.RIGHT:
            if step < last_step:
                next_changes = tape_changed_indices[tape_changed_indices > step]
                if len(next_changes) > 0:
                    step = next_changes[0]
                else:
                    step = last_step

        cols, _ = shutil.get_terminal_size()
        line_count = 0
        for n in lengths:
            for _ in range(0, n, cols):
                line_count += 1

        print(f'\x1b[{line_count}A', end='')


def main():
    # Turing's example TM, makes 0_1_0_1_...
    alternator = make_TM_spec(
        [
            ('b', '_', '0', 'R', 'c'),
            ('c', '_', '_', 'R', 'e'),
            ('e', '_', '1', 'R', 'k'),
            ('k', '_', '_', 'R', 'b'),
        ],
        symbols='01',
        initial_state='b',
        halt_state='halt',
        blank='_',
        left='L',
        right='R',
        stay='N',
    )

    # Doubles the contents of the tape (a string of 0s and 1s)
    doubler = make_TM_spec(
       [
           ('init', '1', 'y', 'R', 'init'),
           ('init', '0', 'x', 'R', 'init'),
           ('init', '_', '_', 'L', 'skip'),

           ('skip', '0', '0', 'L', 'skip'),
           ('skip', '1', '1', 'L', 'skip'),
           ('skip', 'x', 'x', 'L', 'find'),
           ('skip', 'y', 'y', 'L', 'find'),

           ('find', 'x', 'x', 'L', 'find'),
           ('find', 'y', 'y', 'L', 'find'),
           ('find', '0', '0', 'R', 'copy'),
           ('find', '1', '1', 'R', 'copy'),
           ('find', '_', '_', 'R', 'copy'),

           ('copy', 'x', '0', 'R', 'write0'),
           ('copy', 'y', '1', 'R', 'write1'),

           ('write0', '01xy', 'R', 'write0'),
           ('write0', '_', '0', 'L', 'skip'),

           ('write1', '01xy', 'R', 'write1'),
           ('write1', '_', '1', 'L', 'skip'),
       ],
       symbols='01xy',
       initial_state='init',
       halt_state='halt',
       blank='_',
       left="L",
       right="R",
       stay="S"
    )

    # A very simple TM to test the UTM on
    simple = make_TM_spec(
        [
            ('b', '_', '0', 'R', 'c'),
            ('c', '_', '1', 'R', 'c'),
        ],
        symbols='01',
        initial_state='b',
        halt_state='halt',
        blank='_',
        left='L',
        right='R',
        stay='N',
    )

    # print(format_execution(doubler, *run_TM(doubler, "10010", 200)))
    # print(encode(doubler, "_"))

    utm = make_UTM()
    encoded = encode(simple, '_')
    tapes, heads, states = run_TM(utm, encoded, max_steps=2000)

    # Run the TM interactively
    interactive_TM(spec=utm, tapes=tapes, heads=heads, states=states, lower_index=-1)


if __name__ == "__main__":
    main()