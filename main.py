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

@dataclass
class TuringMachineSpec:
    delta: Int[Array, "Q S 3"]
    symbols: List[str]
    states: List[str]

def make_TM_spec(
        transitions: List[tuple[str, str, str, str, str]],
        initial_state: str,
        halt_state: str,
        blank: str,
        left: str,
        right: str,
        stay: str,
    ) -> TuringMachineSpec:
    assert initial_state != halt_state, "Haven't thought about this case"
    symbols = set()
    states = set()
    inputs = set()
    for (read, state, write, next_state, move) in transitions:
        assert read == blank or type(read) == str, f"Bad input symbol {read} in {(read, state, write, next_state, move)}"
        assert write == blank or type(write) == str, f"Bad output symbol {write} in {(read, state, write, next_state, move)}"
        assert move in {left, right, stay}, f"Bad move {move} in {(read, state, write, next_state, move)}"
        if (read,state) in inputs:
            raise ValueError(f"Duplicate input {(read, state)} in transitions")
        inputs.add((read,state))

        if read != blank:
            symbols.add(read)
        if write != blank:
            symbols.add(write)
        if state != initial_state and state != halt_state:
            states.add(state)
        if next_state != initial_state and next_state != halt_state:
            states.add(next_state)

    symbols = [blank] + list(symbols)
    states = [initial_state, halt_state] + list(states)
    symbol_to_int = {s: i for i, s in enumerate(symbols)}
    state_to_int = {q: i for i, q in enumerate(states)}
    move_to_int = {stay: 0, left: 1, right: 2}

    delta = jnp.zeros((len(symbols), len(states), 3), dtype=jnp.uint8)
    for read, state, write, next_state, move in transitions:
        delta = delta.at[symbol_to_int[read], state_to_int[state]].set(jnp.array([symbol_to_int[write], state_to_int[next_state], move_to_int[move]], dtype=jnp.uint8))

    # Set all unspecified transitions to halt and stay
    for i, s in enumerate(symbols):
        for j, q in enumerate(states):
            if (s,q) not in inputs:
                delta = delta.at[i, j].set(jnp.array([i, 1, 0], dtype=jnp.uint8))

    return TuringMachineSpec(delta, symbols, states)

def step(tape: Int[Array, "t"], head: int, state: int, delta: Int[Array, "S Q 3"]) -> tuple[Int[Array, "t"], int, int]:
    read = tape[head]
    write, new_state, move = delta[read, state]
    tape = jnp.where(state == 1, tape, tape.at[head].set(write))
    dir = jnp.where(move == 0, 0, jnp.where(move == 1, -1, 1))
    head = jnp.where(state == 1,head, head + dir)
    state = jnp.where(state == 1, state, new_state)
    return (tape, head, state)

def multi_step(
        start_tape: Int[Array, "t"],
        start_head: int,
        start_state: int,
        delta: Int[Array, "S Q 3"],
        num_steps: int
    ) -> tuple[Int[Array, "h t"], Int[Array, "h"], Int[Array, "h"]]:

    def body(config, _):
        stepped = step(*config, delta)
        return (stepped, stepped)

    _, (tapes, heads, states) = jax.lax.scan(body, (start_tape, start_head, start_state), length=num_steps)
    return tapes, heads, states

def run_TM(spec: TuringMachineSpec, input_string: str, max_steps: int = 1000) -> tuple[str, str]:
    delta = spec.delta

    head_start = max(max_steps, len(input_string))
    init_tape = jnp.zeros(2*head_start+1, dtype=jnp.uint8)
    init_tape = init_tape.at[head_start:head_start+len(input_string)].set(jnp.array([spec.symbols.index(s) for s in input_string], dtype=jnp.uint8))

    tapes, heads, states = multi_step(init_tape, head_start, jnp.astype(0, jnp.uint8), delta, max_steps)
    # Prepend initial values
    tapes = jnp.concatenate([jnp.expand_dims(init_tape, 0), tapes])
    heads = jnp.concatenate([jnp.array([head_start], dtype=jnp.uint32), heads])
    states = jnp.concatenate([jnp.array([0], dtype=jnp.uint8), states])
    return tapes, heads, states

def format_execution(spec: TuringMachineSpec, tapes: Int[Array, "h t"], heads: Int[Array, "h"], states: Int[Array, "h"]) -> str:
    first = jnp.min(heads)
    last = jnp.max(heads)
    
    lines = []
    longest_state_name = max(len(q) for q in spec.states)
    for tape, head, state in zip(tapes, heads, states):
        tape_chars = [spec.symbols[i] for i in tape[first:head]]
        tape_chars.append(f"\033[48;5;240m{spec.symbols[tape[head]]}\033[0m")
        tape_chars.extend([spec.symbols[i] for i in tape[head+1:last+1]])
        lines.append(f"[{spec.states[state]:<{longest_state_name}}] " + "".join(tape_chars))
        if state == 1:
            break

    return "\n".join(lines)

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
    moves = jnp.array([-1, 1])
    # Generate all possible outputs
    outputs = jnp.stack(jnp.meshgrid(symbols, states, moves, indexing='ij'), axis=-1)
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
    for read in range(delta.shape[0]):
        for state in range(delta.shape[1]):
            write, next_state, move = delta[read, state]
            
            # Ignore transitions out of halt state and replace 1 -> n, 0 -> 1
            if state == 0:
                state = 1
            elif state == 1:
                continue

            if next_state == 0:
                next_state = 1
            elif next_state == 1:
                next_state = delta.shape[1]

            move_letter = ["N", "L", "R"][move]
            expressions.append("D" + "A" * (state) + "D" + "C" * read + "D" + "C" * write + move_letter + "D" + "A" * (next_state + 1))
    
    description = ";".join(expressions)

    return "əə" + blank.join(description) + "$"

def make_UTM():
    symbols = 'ACD01uvwxyzə_'
    blank = '_'

    def sequence(reads, state: str, actions, final_state: str):
        '''
        Example: sequence('abc', 'q', ('P1', 'R', 'P0', 'L', 'L'), 'p')
        Behaviour: "In state q if reading 'a', 'b' or 'c', print 1, go right, print 0, then go left twice, and enter state p"
        '''
        next_state = f'{state}\'' if len(actions) > 1 else final_state

        for i, action in enumerate(actions):
            if i == len(actions) - 1:
                next_state = final_state
            else:
                next_state = f'{state}\''
            if action[0] == 'P':
                yield from ((s, state, action[1], next_state, 'N') for s in reads)
            else:
                assert action in "LR"
                yield from ((s, state, s, next_state, action) for s in reads)
            
            reads = symbols

    def switch(reads, state: str, new_state: str):
        '''
        Example: switch('abc', 'q', 'p')
        Behaviour: "In state q if reading 'a', 'b' or 'c', stay in the same place and enter state p"
        '''
        yield from ((s, state, s, new_state, 'N') for s in reads)

    def f(C,B,a):
        q = f'f({C},{B},{a})'
        q1 = f'f1({C},{B},{a})'
        q2 = f'f2({C},{B},{a})'
        yield ('ə', q, 'ə', q1, 'L')
        yield from ((s, q, s, q, 'L') for s in symbols if s != 'ə')

        yield (a, q1, a, C, 'N')
        yield from ((s,q1,s,q1,'R') for s in symbols if s != blank and s != a)
        yield (blank,q1,blank,q2,'R')

        yield (a, q2, a, C, 'N')
        yield from ((s,q2,s,q1,'R') for s in symbols if s != blank and s != a)
        yield (blank,q1,blank,B,'R')


    assert False, "Not done yet"

    return make_TM_spec(
        [
            *((s, 'b', s, 'f(b1,b1,$)') for s in symbols),
            *f('b1','b1','$'),
            *sequence(symbols, 'b1', ('R','R','P:','R','R','PD','R','R','PA'), 'anf'),
            *((s, 'b', s, 'f(b1,b1,$)') for s in symbols),
            # *switch(symbols, 'anf', g('anf1', ':')),
        ]
    )

def main():
    # Doubles the contents of the tape (a string of 0s and 1s)
    doubler = make_TM_spec(
       [
           ('0', 'init', 'x', 'init', 'R'),
           ('1', 'init', 'y', 'init', 'R'),
           ('_', 'init', '_', 'skip', 'L'),

           ('0', 'skip', '0', 'skip', 'L'),
           ('1', 'skip', '1', 'skip', 'L'),
           ('x', 'skip', 'x', 'find', 'L'),
           ('y', 'skip', 'y', 'find', 'L'),

           ('x', 'find', 'x', 'find', 'L'),
           ('y', 'find', 'y', 'find', 'L'),
           ('0', 'find', '0', 'copy', 'R'),
           ('1', 'find', '1', 'copy', 'R'),
           ('_', 'find', '_', 'copy', 'R'),

           ('x', 'copy', '0', 'write0', 'R'),
           ('y', 'copy', '1', 'write1', 'R'),

           *((s, 'write0', s, 'write0', 'R') for s in '01xy'),
           ('_', 'write0', '0', 'skip', 'L'),

           *((s, 'write1', s, 'write1', 'R') for s in '01xy'),
           ('_', 'write1', '1', 'skip', 'L'),

       ],
       initial_state='init',
       halt_state='halt',
       blank='_',
       left="L",
       right="R",
       stay="S"
    )

    print(format_execution(doubler, *run_TM(doubler, "10010", 200)))
    # print(encode(doubler, "_"))


if __name__ == "__main__":
    main()