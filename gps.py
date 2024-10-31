'''
    Geometry of Program Synthesis - experiment recreation

    We propogate uncertainty through a Turing machine, recreating the experiment
    of Clift, Murfet, Wallbridge "Geometry of Program Synthesis".

    We specify transitions in the order (s,q,s',q',d), to match the paper,
    contrasting the (q,s,s',d,q') order that is used in other files (which
    reads more naturally imo). There is also no halt_state.

    # Internal Representation
    Symbols, states, directions are integers

    0 is the blank symbol
    0 is the initial state
    0 is stay (no move)
    1 is left
    2 is right
'''

from dataclasses import dataclass
from jaxtyping import Array, Int, Float, Bool, PRNGKeyArray
import jax
import jax.numpy as jnp
from typing import List, Tuple
import numpy as np
import readchar
import shutil
import functools
import einops

@dataclass
class SmoothTuringMachine:
    tape_alphabet: List[str]
    states: List[str]
    moves: List[str]
    delta_write: Float[Array, "Σ Q Σ"]
    delta_state: Float[Array, "Σ Q Q"]
    delta_move: Float[Array, "Σ Q 3"]

    def theta_from_input_pairs(
        self,
        pairs: List[Tuple[str,str]]
    ) -> Float[Array, "T 2"]:
        '''
        Convert (symbol,state) pairs (as strings) to an array of indices that
        can be used to index into arrays of shape (Σ,Q).
        '''
        assert len(set(pairs)) == len(pairs), "pairs must be unique"
        return jnp.array([(self.tape_alphabet.index(s), self.states.index(q)) for (s, q) in pairs])
    
    def descriptions_from_theta(
        self,
        theta: Int[Array, "T 2"] | None,
    ) -> Tuple[Int[Array, "T Σ"], Int[Array, "T Q"], Int[Array, "T 3"]]:
        '''
        Generate the TM "description" for use in direct-simulation.
        Converts the delta arrays from (Σ,Q)-indexing to (T,)-indexing via theta
        '''
        if theta is None:
            return self.delta_write.ravel(), self.delta_state.ravel(), self.delta_move.ravel()
        descr_write = self.delta_write[theta[:,0], theta[:,1]]
        descr_state = self.delta_state[theta[:,0], theta[:,1]]
        descr_move  = self.delta_move[theta[:,0],  theta[:,1]]
        return descr_write, descr_state, descr_move
    
    def prepare_initial_config(
        self,
        input_symbols: List[str],
        tape_radius: int,
    ) -> Tuple[Int[Array, "t Σ"], Int[Array, "Q"], Int]:
        
        Σ = len(self.tape_alphabet)
        Q = len(self.states)

        blank = jax.nn.one_hot(0, Σ)
        head_zero = tape_radius

        # Initialise tape to all blanks, then write string from head_zero onward
        start_tape = jnp.broadcast_to(blank, (tape_radius*2+1, Σ))
        start_tape = start_tape.at[head_zero:head_zero+len(input_symbols)].set(
            jax.nn.one_hot(
                jnp.array([self.tape_alphabet.index(s) for s in input_symbols]),
                Σ
            )
        )

        # Begin with certainty of the initital state
        start_state = jax.nn.one_hot(0, Q)
        
        return start_tape, start_state, head_zero
    
    @functools.partial(jax.jit, static_argnames=['input_symbols', 'tape_radius', 'num_steps', 'theta'])
    def run(
        self,
        input_symbols: List[str],
        tape_radius: int,
        num_steps: int,
        theta: Int[Array, "T 2"] | None = None,
    ) -> tuple[Int[Array, "t Σ"], Int[Array, "Q"], Int]:
        assert tape_radius - 1 >= len(input)

        descr_write, descr_state, descr_move = self.descriptions_from_theta(theta)
        start_tape, start_state, head_zero = self.prepare_initial_config(input_symbols, tape_radius)

        def iter_func(_, config):
            return direct_sim_step(
                *config,
                head_zero=head_zero,
                descr_write=descr_write,
                descr_state=descr_state,
                descr_move=descr_move,
                theta=theta,
            )
        
        final_tape, final_state = jax.lax.fori_loop(
            lower=0,
            upper=num_steps,
            body=iter_func,
            init_val=(start_tape, start_state)
        )

        return final_tape, final_state, head_zero

    # @functools.partial(jax.jit, static_argnames=['input_symbols', 'tape_radius', 'num_steps'])
    def run_history(
        self,
        input_symbols: List[str],
        tape_radius: int,
        num_steps: int,
        theta: Int[Array, "T 2"] | None = None,
    ) -> tuple[Int[Array, "h t Σ"], Int[Array, "h Q"], Int]:
        assert tape_radius - 1 >= len(input_symbols)

        descr_write, descr_state, descr_move = self.descriptions_from_theta(theta)
        start_tape, start_state, head_zero = self.prepare_initial_config(input_symbols, tape_radius)

        def scan_func(config, _):
            stepped = direct_sim_step(
                *config,
                head_zero=head_zero,
                descr_write=descr_write,
                descr_state=descr_state,
                descr_move=descr_move,
                theta=theta,
            )
            return (stepped, config)

        _, (tape_history, state_history) = jax.lax.scan(
            f=scan_func,
            init=(start_tape, start_state),
            length=num_steps+1,
        )
        return tape_history, state_history, head_zero
    
    def nudge_write(
        self,
        symbol: str,
        state: str,
        target_write: str,
        epsilon: float = 0.01,
    ):
        s, q = self.tape_alphabet.index(symbol), self.states.index(state)
        write_dist = self.delta_write[s,q]

        new_delta_write = self.delta_write.at[s, q].set(
            (1-epsilon) * write_dist + epsilon * jax.nn.one_hot(self.tape_alphabet.index(target_write), len(self.tape_alphabet))
        )
        return self.replace(delta_write=new_delta_write)

    def nudge_next_state(
        self,
        symbol: str,
        state: str,
        target_next_state: str,
        epsilon: float = 0.01,
    ):
        s, q = self.tape_alphabet.index(symbol), self.states.index(state)
        state_dist = self.delta_state[s,q]

        new_delta_state = self.delta_state.at[s, q].set(
            (1-epsilon) * state_dist + epsilon * jax.nn.one_hot(self.states.index(target_next_state), len(self.states))
        )
        return self.replace(delta_state=new_delta_state)

    def nudge_move(
        self,
        symbol: str,
        state: str,
        target_move: str,
        epsilon: float = 0.01,
    ):
        s, q = self.tape_alphabet.index(symbol), self.states.index(state)
        move_dist = self.delta_move[s,q]

        new_delta_move = self.delta_move.at[s, q].set(
            (1-epsilon) * move_dist + epsilon * jax.nn.one_hot(self.moves.index(target_move), len(self.moves))
        )
        return self.replace(delta_move=new_delta_move)


def direct_sim_step(
        tape: Int[Array, "t Σ"],
        state: Int[Array, "Q"],
        head_zero: int,
        descr_write: Int[Array, "T Σ"],
        descr_state: Int[Array, "T Q"],
        descr_move:  Int[Array, "T 3"],
        theta: Int[Array, "T 2"] | None = None,
    ) -> Tuple[Int[Array, "t Σ"], Int[Array, "Q"]]:
    '''
    Direct simulation of the working tape and state of a TM running on a probabilistic UTM
    '''
    T = theta.shape[0]
    Σ = tape.shape[1]
    Q = state.shape[0]

    # head_and_state[σ,q] is the probability that the head is at symbol σ and the state is q
    head_and_state = jnp.outer(tape[head_zero], state)
    assert head_and_state.shape == (Σ, Q)

    # λ[i] is the probability that the ith tuple matches the current head and state
    # λ.shape == (T,)
    if theta is None:
        λ = head_and_state.ravel()
    else:
        λ = head_and_state[theta[:,0], theta[:,1]]

    # μ[i] is the probability that the ith tuple will be written on the staging tape
    # after the scan phase. In other words, it is the probability that the ith tuple
    # is the *last* matching tuple, calculated as λ[i] × Π_{j>i} (1-λ[j])
    # μX is the probability that no tuple is written on the staging tape (so it's stays XXX)
    μX, μ = jax.lax.scan(lambda c, x: ((1-x) * c, x * c), 1, λ, reverse=True)
    assert μ.shape == (T,)
    assert μX.shape == ()

    # These distributions describe what will be on the staging tape after the scan phase
    stage_write = jnp.einsum('is,i->s', descr_write, μ)
    stage_state = jnp.einsum('iq,i->q', descr_state, μ)
    stage_move = jnp.einsum('im,i->m', descr_move, μ)
    assert stage_write.shape == (Σ,)
    assert stage_state.shape == (Q,)
    assert stage_move.shape == (3,)

    move = stage_move + jnp.array([μX, 0, 0])
    write = μX * tape[head_zero] + stage_write # write[σ] is Aσ
    new_state = μX * state + stage_state
    assert move.shape == (3,)
    assert write.shape == (Σ,)
    assert new_state.shape == (Q,)

    new_tape = tape.at[head_zero].set(write)

    blank = jax.nn.one_hot(0, Σ)
    stay, left, right = move[0], move[1], move[2]

    new_tape = (
        stay * new_tape
        # shift tape left and insert blank at end
        + left * jnp.roll(new_tape, 1, axis=0).at[0].set(blank)
        # shift tape right and insert blank at start
        + right * jnp.roll(new_tape, -1, axis=0).at[-1].set(blank)
    )
    assert new_tape.shape == tape.shape

    # Renormalize
    new_tape = new_tape / new_tape.sum(axis=-1, keepdims=True)

    return new_tape, new_state

@dataclass
class TuringMachine:
    # First character is treated as the blank character
    tape_alphabet: List[str]
    # First state is treated as the initial state
    states: List[str]
    # moves = [stay, left, right]
    moves: List[str]
    # tuple order is read, state, write, next_state, move
    transitions: List[Tuple[str, str, str, str, str]]

    def relax(self) -> SmoothTuringMachine:
        stay, left, right = self.moves
        inputs = set()
        
        Σ = len(self.tape_alphabet)
        Q = len(self.states)

        delta_write = np.zeros((Σ, Q, Σ), dtype=np.float32)
        delta_state = np.zeros((Σ, Q, Q), dtype=np.float32)
        delta_move = np.zeros((Σ, Q, 3), dtype=np.float32)

        for read, state, write, next_state, move in self.transitions:
            assert read in self.tape_alphabet, f"Input symbol {read} in {(read, state, write, next_state, move)} not found in tape_alphabet"
            assert write in self.tape_alphabet, f"Output symbol {write} in {(read, state, write, next_state, move)} not found in tape_alphabet"
            assert move in {stay, left, right}, f"Bad move {move} in {(read, state, write, next_state, move)}"
            if (read,state) in inputs:
                raise ValueError(f"Duplicate input {(state, read)} in transitions")
            inputs.add((read,state))

            r = self.tape_alphabet.index(read)
            q = self.states.index(state)
            w = self.tape_alphabet.index(write)
            p = self.states.index(next_state)
            m = self.moves.index(move)

            delta_write[r,q,w] = 1.0
            delta_state[r,q,p] = 1.0
            delta_move[r,q,m] = 1.0
        
         # Set all unspecified transitions to just self-loop
        for i, s in enumerate(self.tape_alphabet):
            for j, q in enumerate(self.states):
                if (s,q) not in inputs:
                    # write what you read
                    delta_write[i,j,i] = 1.0
                    # stay in the same state
                    delta_state[i,j,j] = 1.0
                    # stay in the same place
                    delta_move[i,j,0] = 1.0

        return SmoothTuringMachine(
            tape_alphabet=self.tape_alphabet,
            states=self.states,
            moves=self.moves,
            delta_write=jnp.array(delta_write),
            delta_state=jnp.array(delta_state),
            delta_move=jnp.array(delta_move),
        )

    def transition_input_pairs(self) -> List[Tuple[str,str]]:
        return [(read,state) for (read, state, _, _, _) in self.transitions]


def interactive_TM(
        smooth_tm: SmoothTuringMachine,
        tape_history: Float[Array, "h t Σ"],
        state_history: Float[Array, "h Q"],
        head_zero: int,
    ):

    def bg_greyscale(val: int) -> str:
        '''
        Return a highlight symbol with ANSI codes
        0 <= val <= 255
        '''
        bg = f'\033[48;2;{val};{val};{val}m'
        fg = f'\033[38;2;{0};{0};{0}m'
        return bg+fg

    h, tape_size, Σ = tape_history.shape
    Q = state_history.shape[1]

    greyscale_codes = np.char.array([bg_greyscale(i) for i in range(256)])

    # tapes_by_symbol = einops.rearrange(tape_history, "h t Σ -> h Σ t")
    tapes_greyscale = greyscale_codes[jnp.astype(255 * tape_history, jnp.uint8)]
    states_greyscale = greyscale_codes[jnp.astype(255 * state_history, jnp.uint8)]

    tapes_highlighted = tapes_greyscale + np.char.array(smooth_tm.tape_alphabet)
    states_highlighted = states_greyscale + np.char.array(smooth_tm.states)

    assert tapes_highlighted.shape == (h, tape_size, Σ)
    assert states_highlighted.shape == (h, Q)

    RESET_FG = '\033[39m'
    RESET_BG = '\033[49m'

    # We sum (string concatenate) along the tape-squares axis
    tapes_rendered = [
        [
            ''.join(symbol_tape) + RESET_FG + RESET_BG for symbol_tape in tapes_by_symbol
        ]
        for tapes_by_symbol in einops.rearrange(tapes_highlighted, "h t Σ -> h Σ t")
    ]
    states_rendered = [
        ''.join(state) + RESET_FG + RESET_BG for state in states_highlighted
    ]

    tape_widths = [tape_size * len(symbol) for symbol in smooth_tm.tape_alphabet]
    state_width = sum(len(state) for state in smooth_tm.states)

    step = 0
    while True:

        state = states_rendered[step]
        tape_by_symbol = tapes_rendered[step]

        lines = [
            f'Step: {step}/{h-1} | Press Q to quit | ↓/↑ to step forward/back',
            'State: ' + state,
            'Tape (head=▾):',
            ' ' * head_zero + '▾',
            *tape_by_symbol,
        ]

        # The width of the tapes is complicated by the ANSI codes for the head
        widths = list(map(len,lines))
        widths[2] = len('State: ') + state_width
        widths[4:] = tape_widths

        for line in lines:
            print('\033[2K\r' + line)

        key = readchar.readkey()

        if key.lower() == 'q':
            break
        elif key == readchar.key.UP:
            if step > 0:
                step -= 1
        elif key == readchar.key.DOWN:
            if step < h-1:
                step += 1
        
        cols, _ = shutil.get_terminal_size()
        line_count = 0
        for n in widths:
            for _ in range(0, n, cols):
                line_count += 1

        print(f'\033[{line_count}A', end='')

def main():
    simple = TuringMachine(
        tape_alphabet=list('_01'),
        states=list('bc'),
        moves = list('SLR'),
        transitions=[
            ('_', 'b', '0', 'c', 'R'),
            ('_', 'c', '1', 'b', 'R'),
        ],
    )

    detectA = TuringMachine(
        tape_alphabet=list('_AB'),
        states=list('rq'), # 'r' for reject, 'q' for accept
        moves=list('SLR'),
        transitions=[
            ('_', 'r', '_', 'r', 'S'),  # see a blank, freeze
            ('_', 'q', '_', 'q', 'S'),
            ('A', 'r', 'A', 'q', 'S'),  # see an A, accept and freeze
            ('A', 'q', 'A', 'q', 'S'),
            ('B', 'r', 'B', 'r', 'R'),  # see a B, move right
            ('B', 'q', 'B', 'q', 'S')
        ]
    )

    input_symbols = list('BBBBBBBBBBBBBBBBABA')

    smooth_tm = detectA.relax()
    smooth_tm = smooth_tm.nudge_move(
        symbol='B',
        state='r',
        target_move='S',
        epsilon=0.4
    )
    theta = smooth_tm.theta_from_input_pairs(detectA.transition_input_pairs())

    tape_history, state_history, head_zero = smooth_tm.run_history(
        input_symbols=input_symbols,
        tape_radius=len(input_symbols) + 3,
        num_steps=42,
        theta=theta,
    )

    interactive_TM(smooth_tm, tape_history, state_history, head_zero)

if __name__ == "__main__":
    main()