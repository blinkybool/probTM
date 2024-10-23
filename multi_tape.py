'''
    Multi-tape Turing Machines.

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
import itertools

@dataclass
class TuringMachineSpec:
    num_tapes: int
    delta: Int[Array, "Q ... O"]
    tape_alphabet: List[str]
    states: List[str]

def make_TM_spec(
        num_tapes: int,
        transitions: List[tuple[str, ...]],
        symbols: List[str],
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
    symbols = list(symbols)
    inputs = set()

    # converted = []
    # for transition in transitions:
    #     if len(transition) == 5:
    #         converted.append(transition)
    #     elif len(transition) == 4:
    #         state, reads, action, next_state = transition
    #         assert all(read in symbols+blank for read in reads)
    #         if action in {stay, left, right}:
    #             for read in reads:
    #                 converted.append((state, read, read, action, next_state))
    #         else:
    #             assert len(action) == 2, f"Bad action {action}"
    #             assert action[0] == 'P', f"Bad action {action}"
    #             assert action[1] == blank or action[1] in symbols, f"Bad action {action}"
    #             for read in reads:
    #                 converted.append((state, read, action[1], stay, next_state))
    #     else:
    #         raise ValueError(f"Bad transition {transition}")

    tape_alphabet = [blank] + symbols


    for transition in transitions:
        assert len(transition) == 5, f"Bad transition {transition} (only {len(transition)} entries)"
        state, reads, writes, moves, next_state = transition
        assert len(reads) == num_tapes, f"Bad transition {transition=}, reads={reads} but {num_tapes=}"
        assert len(writes) == num_tapes, f"Bad transition {transition=}, writes={writes} but {num_tapes=}"
        assert len(moves) == num_tapes, f"Bad transition {transition=}, moves={writes} but {num_tapes=}"
        assert set(reads) <= set(tape_alphabet), f"Bad {transition=}, {reads=} not all found in {tape_alphabet=}"
        assert set(writes) <= set(tape_alphabet), f"Bad {transition=}, {writes=} not all found in {tape_alphabet=}"
        assert set(moves) <= {left, right, stay}, f"Bad {transition=}, {moves=} not all valid ({left + right + stay})"
        if (state,tuple(reads)) in inputs:
            raise ValueError(f"Duplicate input {(state, reads)} in transitions")
        inputs.add((state,tuple(reads)))

        if state != initial_state and state != halt_state and state not in states:
            states.append(state)
        if next_state != initial_state and next_state != halt_state and next_state not in states:
            states.append(next_state)

    states = [initial_state, halt_state] + list(states)
    symbol_to_int = {s: i for i, s in enumerate(tape_alphabet)}
    state_to_int = {q: i for i, q in enumerate(states)}
    move_to_int = {stay: 0, left: 1, right: 2}

    shape = (len(states),) + num_tapes * (len(tape_alphabet),) + (2 * num_tapes + 1,)
    delta = np.zeros(shape, dtype=np.uint8)
    for state, reads, writes, moves, next_state in transitions:
        state = state_to_int[state]
        reads = tuple(symbol_to_int[read] for read in reads)
        writes = tuple(symbol_to_int[write] for write in writes)
        moves = tuple(move_to_int[move] for move in moves)
        next_state = state_to_int[next_state]

        input = (state,) + reads
        output = np.array(writes + moves + (next_state,), dtype=np.uint8)
        delta[input] = output

    # Set all unspecified transitions to halt and stay
    for q, i in state_to_int.items():
        for pairs in itertools.product(symbol_to_int.items(), repeat=num_tapes):
            reads = tuple(s for s, j in pairs)
            js = tuple(j for s, j in pairs)
            if (q,reads) not in inputs:
                input = (i,) + js
                output = np.array(js + (0,) * num_tapes + (state_to_int[halt_state],), dtype=np.uint8)
                delta[input] = output

    return TuringMachineSpec(num_tapes, delta, tape_alphabet, states)

def step(multi_tape: Int[Array, "m t"], multi_heads: Int[Array, "m"], state: int, delta: Int[Array, "Q ... O"]) -> tuple[Int[Array, "m t"], Int[Array, "m"], int]:
    reads = multi_tape[jnp.arange(multi_tape.shape[0]), multi_heads]
    # jax.debug.print("state={state} reads={reads}", state=state, reads=reads)
    output = delta[state, *reads]
    # jax.debug.print("output={output}", output=output)
    writes = output[:multi_tape.shape[0]]
    # jax.debug.print("writes={writes}", writes=writes)
    moves = output[multi_tape.shape[0]:-1]
    # jax.debug.print("moves={moves}", moves=moves)
    new_state = output[-1]
    # jax.debug.print("new_state={new_state}", new_state=new_state)
    multi_tape  = jnp.where(state == 1, multi_tape,  multi_tape.at[jnp.arange(multi_tape.shape[0]), multi_heads].set(writes))
    dirs = jnp.where(moves == 0, 0, jnp.where(moves == 1, -1, 1))
    multi_heads = jnp.where(state == 1, multi_heads, multi_heads + dirs)
    new_state = jnp.where(state == 1, state, new_state)
    return (multi_tape, multi_heads, new_state)

def iter_step(
        start_multi_tape: Int[Array, "m t"],
        start_multi_head: Int[Array, "m"],
        start_state: int,
        delta: Int[Array, "Q ... O"],
        num_steps: int
    ) -> tuple[Int[Array, "h m t"], Int[Array, "h m"], Int[Array, "h"]]:

    def body(config, _):
        stepped = step(*config, delta)
        return (stepped, stepped)

    _, (multi_tapes, multi_heads, multi_states) = jax.lax.scan(body, (start_multi_tape, start_multi_head, start_state), length=num_steps)
    return multi_tapes, multi_heads, multi_states

def run_TM(
        spec: TuringMachineSpec,
        multi_input,
        multi_tape_shape,
        head_zeros,
        max_steps: int = 1000
    ) -> tuple[Int[Array, "h m t"], Int[Array, "h m"], Int[Array, "h"]]:
    assert len(multi_input) == spec.num_tapes

    delta = jnp.array(spec.delta)
    init_multi_head = jnp.broadcast_to(head_zeros, (spec.num_tapes,)).astype(jnp.uint32)
    init_multi_tape = jnp.zeros(multi_tape_shape, dtype=jnp.uint8)
    for i, input_string in enumerate(multi_input):
        input_array = jnp.array([spec.tape_alphabet.index(s) for s in input_string], dtype=jnp.uint8)
        init_multi_tape = init_multi_tape.at[i, 0:len(input_array)].set(input_array)

    multi_tapes, multi_heads, states = jax.jit(iter_step, static_argnames='num_steps')(init_multi_tape, init_multi_head, jnp.astype(0, jnp.uint8), delta, max_steps)
    # Prepend initial values
    multi_tapes = jnp.concatenate([jnp.expand_dims(init_multi_tape, 0), multi_tapes])
    multi_heads = jnp.concatenate([jnp.expand_dims(init_multi_head, 0), multi_heads])
    states = jnp.concatenate([jnp.array([0], dtype=jnp.uint8), states])
    return multi_tapes, multi_heads, states

def interactive_TM(
        spec,
        multi_tapes,
        multi_heads,
        states,
    ):
    halt_step = jnp.argmax(states == 1)
    if states[halt_step] == 1:
        last_step = halt_step
    else:
        last_step = len(multi_tapes)-1

    changed_indices = jnp.where(jnp.any(multi_tapes[1:] != multi_tapes[:-1], axis=1))[0] + 1

    as_symbols = np.array(list(spec.tape_alphabet))[multi_tapes]
    as_symbol_lens = np.array(list(len(s) for s in spec.tape_alphabet), dtype=np.uint8)[multi_tapes]
    
    rendered = [
        [
            ''.join(tape[0:head])
            + f"\033[48;5;240m{tape[head]}\033[0m"
            + ''.join(tape[head+1:])
            if head < len(tape) else ''.join(tape)
            for tape, head in zip(multi_tape, multi_head)
        ]
        for multi_tape, multi_head in zip(as_symbols, multi_heads)
    ]


    step = 0
    while True:

        state = states[step]
        multi_tape = rendered[step]
        multi_tape_widths = as_symbol_lens[step].sum(axis=-1)

        lines = [
            f"Step: {step}/{last_step} | Press Q to quit",
            f"State: {spec.states[state]}",
            *multi_tape
        ]

        # The width of the tapes is complicated by the ANSI codes for the head
        widths = [len(lines[0]), len(lines[1])] + list(multi_tape_widths)

        for line in lines:
            print('\33[2K\r' + line)

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
                prev_changes = changed_indices[changed_indices < step] - 1
                if len(prev_changes) > 0:
                    step = prev_changes[-1]
                else:
                    step = 0
        elif key == readchar.key.RIGHT:
            if step < last_step:
                next_changes = changed_indices[changed_indices > step]
                if len(next_changes) > 0:
                    step = next_changes[0]
                else:
                    step = last_step
        

        cols, _ = shutil.get_terminal_size()
        line_count = 0
        for n in widths:
            for _ in range(0, n, cols):
                line_count += 1

        print(f'\x1b[{line_count}A', end='')

def make_pseudo_utm(
    sim_states: List[str],
    sim_symbols: List[str],
    sim_blank: str = '_',
    utm_blank: str = '□',
):
    
    assert sim_blank != utm_blank, f"Don't want {sim_blank=} same as {utm_blank=}"
    assert sim_blank not in sim_symbols, f"Don't want {sim_blank=} in {sim_symbols=}"
    assert utm_blank not in sim_symbols, f"Don't want {utm_blank=} in {sim_symbols=}"
    assert all(s not in "XLRS" for s in sim_symbols), f"Don't want X,L,R,S in {sim_symbols=}"

    sim_alph = [sim_blank] + list(sim_symbols)
    utm_symbols = ['X', 'L', 'R', 'S'] + sim_alph + sim_states
    utm_alph = [utm_blank] + utm_symbols
    num_tapes = 4
    transitions = []

    notX = [s for s in utm_alph if s != 'X']

    inputs = itertools.product(utm_alph, utm_alph, notX, notX)
    for reads in inputs:
        (a,b,c,d) = reads

        '''
        Update phase
        '''

        # compSymbol (modified from GPS paper)
        if a == 'X':
            # If we've reached the end of the description, then the staging
            # head is not X if we found what we have to do next
            if b != 'X':
                transitions.append(('compSymbol', reads, reads, 'LLSS', 'updateSymbol'))
        else:
            # Otherwise we're looking for a transition matching the current working head symbol
            # and state on the state tape (we'll go to compState to check that)
            if a == d:
                transitions.append(('compSymbol', reads, reads, 'RLSS', 'compState'))
            elif a == sim_blank and d == utm_blank:
                transitions.append(('compSymbol', reads, (a,b,c,sim_blank), 'RLSS', 'compState'))
            else:
                transitions.append(('compSymbol', reads, reads, 'RLSS', '¬compState'))

        # Here we do one step of the simulated TM if we have something on the staging tape, or
        # nothing if we don't.
        if a != 'X':
            if b == 'X':
                # Left path of update phase (do nothing)
                transitions.append(('updateSymbol', reads, reads, 'SRSS', 'updateState'))
                transitions.append(('updateState', reads, reads, 'SRSS', 'updateDir'))
                transitions.append(('updateDir', reads, reads, 'SRSS', 'resetDescr'))
            else:
                # Right path (do simulation for one step)
                transitions.append(('updateSymbol', reads, (a,'X',c,b), 'SRSS', 'updateState'))
                transitions.append(('updateState', reads, (a,'X',b,d), 'SRSS', 'updateDir'))
                if b in "LRS":
                    transitions.append(('updateDir', reads, (a,'X',c,d), 'SLS' + b, 'resetDescr'))

        # resetDescr
        if a != 'X': # removed `and b != 'X'`
            transitions.append(('resetDescr', reads, reads, 'LSSS', 'resetDescr'))
        else: # removed `a == 'X' and b != 'X'`
            transitions.append(('resetDescr', reads, reads, 'RSSS', 'compSymbol'))

        '''
        scan phase
        '''

        # compState
        if a == c and a != 'X': # removed `and b != 'X'`
            transitions.append(('compState', reads, reads, 'RSSS', 'copySymbol'))
        else:
            transitions.append(('compState', reads, reads, 'RSSS', '¬copySymbol'))

        if a != 'X': # removed `and b != 'X'`
            # Copy ouputs of transition to staging tape
            transitions.append(('copySymbol', reads, (a,a,c,d), 'RRSS', 'copyState'))
            transitions.append(('copyState', reads, (a,a,c,d), 'RRSS', 'copyDir'))
            transitions.append(('copyDir', reads, (a,a,c,d), 'RLSS', 'compSymbol'))

            # Just walk through the transition if we decided it doesn't match
            transitions.append(('¬compState', reads, reads, 'RSSS', '¬copySymbol'))
            transitions.append(('¬copySymbol', reads, reads, 'RRSS', '¬copyState'))
            transitions.append(('¬copyState', reads, reads, 'RRSS', '¬copyDir'))
            transitions.append(('¬copyDir', reads, reads, 'RLSS', 'compSymbol'))

    return make_TM_spec(
        num_tapes=num_tapes,
        transitions=transitions,
        symbols=utm_symbols, # exclude utm_blank
        initial_state='compSymbol',
        halt_state='halt',
        blank=utm_blank,
        left='L',
        right='R',
        stay='S',
    )

def encode_description(transitions):
    descr = ['X']
    for state, read, write, move, next_state in transitions:
        descr.extend([read, state, write, next_state, move])
    descr.append('X')
    return descr

def main():

    # Doubles the contents of the tape (a string of 0s and 1s)
    doubler = make_TM_spec(
        num_tapes=2,
        transitions=[
           ('copy', '0_', '00', 'RR', 'copy'),
           ('copy', '1_', '11', 'RR', 'copy'),
           ('copy', '__', '__', 'SL', 'return'),
           ('return', '_0', '_0', 'SL', 'return'),
           ('return', '_1', '_1', 'SL', 'return'),
           ('return', '__', '__', 'SR', 'extend'),
           ('extend', '_0', '00', 'RR', 'extend'),
           ('extend', '_1', '11', 'RR', 'extend'),
           ('extend', '__', '__', 'SS', 'halt'),
       ],
       symbols='01xy',
       initial_state='copy',
       halt_state='halt',
       blank='_',
       left="L",
       right="R",
       stay="S"
    )

    simple_transitions = [
        ('b', '_', '0', 'R', 'c'),
        ('c', '_', '1', 'R', 'b'),
    ]

    doubler_transitions = [
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

        *[('write0', s, s, 'R', 'write0') for s in '01xy'],
        ('write0', '_', '0', 'L', 'skip'),

        *[('write1', s, s, 'R', 'write1') for s in '01xy'],
        ('write1', '_', '1', 'L', 'skip'),
    ]

    encoded = encode_description(doubler_transitions)

    utm = make_pseudo_utm(
        sim_states=['init', 'skip', 'find', 'copy', 'write0', 'write1'],
        sim_symbols=list('01xy'),
        sim_blank='_',
    )

    multi_tapes, multi_heads, states = run_TM(
        spec=utm,
        multi_input=(encoded, ['X','X','X'], ['init'], list('_01001')),
        multi_tape_shape=((4,128)),
        head_zeros=jnp.array([1, 1, 0, 1]),
        max_steps=245 * 100,
    )

    # Run the TM interactively
    interactive_TM(spec=utm, multi_tapes=multi_tapes, multi_heads=multi_heads, states=states)


if __name__ == "__main__":
    main()