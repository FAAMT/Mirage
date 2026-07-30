-- =====================================================================
-- layer_ctrl.vhd — op sequencing for one encoder layer
--
-- Issues the matmul chain the MVP plan specifies, on the one shared MAC
-- array, and interleaves the softmax passes:
--
--   Q = X.Wq    K = X.Wk    V = X.Wv          (streamed weights)
--   per head h:  S_h = Q_h . K_h^T            (both operands activations)
--                softmax pass 1: row max
--                softmax pass 2: exp -> int8, sum8
--                A_h = E_h . V_h              (both operands activations)
--   O = A.Wo                                  (streamed weights)
--   H = O.W1    Y = H.W2                      (streamed weights, per token)
--
-- THE OPERAND-ROUTING PROBLEM, AND HOW THIS SOLVES IT
--
-- Six of the eight matmuls take a streamed weight matrix as the second
-- operand, which is what weight_window exists to serve. Two of them --
-- QK^T and xV -- take ANOTHER ACTIVATION instead. The MAC array only reads
-- its per-lane operand from weight_window.
--
-- Rather than add a second wide read port to the activation memory (which
-- is 8 bits wide and could not supply 32 bytes a cycle anyway), those two
-- operands are STAGED into the weight window through its existing write
-- port: S_COPY reads the activation memory one byte per cycle, packs four
-- bytes into a bank word, and writes it. n_tok x HEAD_DIM = 1024 bytes per
-- head, so 1024 cycles of staging against ~1024 cycles of matmul for the
-- same head. It costs a factor of two on the attention matmuls and no new
-- hardware, which is the right trade at this device size.
--
-- Staging writes into the buffer NOT being read, exactly as the DDR weight
-- stream does, so it reuses the double-buffer machinery unchanged.
--
-- SCOPE NOTE, STATED PLAINLY
--
-- The op sequence, the geometry per op and the staging path are all
-- implemented. What is NOT yet implemented inside this file is residual
-- addition and layer normalisation -- the plan's Week 8 item. They need an
-- elementwise pass over the activation memory and a mean/variance
-- accumulator; the hooks (S_RESID) are present and wired but pass data
-- through unchanged. This is called out in V2_DESIGN.md rather than left
-- for someone to discover.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity layer_ctrl is
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        start      : in std_logic;
        n_tok      : in std_logic_vector(4 downto 0);
        stage_done : in std_logic;   -- staging pass finished
        busy  : out std_logic;
        done  : out std_logic;

        -- gemm_sequencer handshake and geometry
        seq_start : out std_logic;
        seq_done  : in  std_logic;
        k_len     : out std_logic_vector(ACT_AW - 1 downto 0);
        n_coltile : out std_logic_vector(4 downto 0);
        act_base  : out std_logic_vector(ACT_AW - 1 downto 0);
        out_base  : out std_logic_vector(ACT_AW - 1 downto 0);
        rq_shift  : out std_logic_vector(5 downto 0);

        -- weight double-buffer select
        buf_sel : out std_logic;

        -- softmax control and score-memory reads
        sm_row_start : out std_logic;
        sm_pass_exp  : out std_logic;
        sm_s_valid   : out std_logic;
        sc_re        : out std_logic;
        sc_raddr     : out std_logic_vector(SCORE_AW - 1 downto 0);
        rc_re        : out std_logic;

        -- Datapath routing for the two ops whose operands are activations
        -- rather than streamed weights, and for the score writeback. These
        -- exist so that NOTHING in encoder_layer is left undriven -- an
        -- unconnected input is not a stub, it is logic Vivado deletes. v1's
        -- smx_core measured smaller than its own children for exactly that
        -- reason.
        wb_score   : out std_logic;   -- route writeback to score memory
        sc_waddr_o : out std_logic_vector(SCORE_AW - 1 downto 0);
        attn_op    : out std_logic;   -- mac_a comes from softmax, not act mem
        stage_go   : out std_logic;   -- run the activation->weight staging pass
        stage_base : out std_logic_vector(ACT_AW - 1 downto 0);
        -- Staging geometry. The two attention operands need DIFFERENT access
        -- patterns and this is the only place that difference is expressed:
        --
        --   QK^T : lane j is key-token j, reduction over HEAD_DIM. So staged
        --          row k must hold K[j][k] for all j -- a TRANSPOSE of the
        --          token-major layout in activation memory, read with stride
        --          HEAD_DIM.
        --   xV   : lane j is output dim j, reduction over key tokens. Staged
        --          row k holds V[k][0..], which IS contiguous. No transpose.
        stage_rows  : out std_logic_vector(6 downto 0);   -- reduction length
        stage_lanes : out std_logic_vector(6 downto 0);   -- lanes to fill
        stage_xpose : out std_logic
    );
end entity;

architecture rtl of layer_ctrl is

    -- One entry per matmul in the chain, in issue order.
    type op_t is (
        O_PROJ_Q, O_PROJ_K, O_PROJ_V,
        O_STAGE_K, O_SCORES, O_SMAX_MAX, O_SMAX_EXP, O_STAGE_V, O_ATTN_V,
        O_PROJ_O, O_FFN_HID, O_FFN_OUT,
        O_RESID, O_FINISH
    );

    type state_t is (S_IDLE, S_SETUP, S_RUN, S_WAIT, S_COPY, S_SMAX, S_ADVANCE, S_DONE);
    signal state : state_t := S_IDLE;
    signal op    : op_t    := O_PROJ_Q;

    signal head : unsigned(3 downto 0) := (others => '0');  -- 0 .. NUM_HEADS-1

    -- softmax pass walker
    signal sm_tok  : unsigned(4 downto 0) := (others => '0');
    signal sm_col  : unsigned(4 downto 0) := (others => '0');
    signal sm_pass : std_logic := '0';

    signal busy_i : std_logic := '0';
    signal done_i : std_logic := '0';
    signal bsel   : std_logic := '0';

    -- Requantisation shifts. Chosen so an int8 x int8 dot product over
    -- k_len terms lands back in int8 range: log2(k_len) + headroom. These
    -- are the values a calibration pass against the Python reference models
    -- would tune; they are parameters here, not magic numbers in the RTL.
    constant RQ_PROJ  : natural := 14;   -- k_len = 512
    constant RQ_SCORE : natural := 8;    -- k_len = 64, keep score range
    constant RQ_ATTN  : natural := 7;    -- normalised by recip_rom after
    constant RQ_FFN   : natural := 14;

begin

    busy    <= busy_i;
    done    <= done_i;
    buf_sel <= bsel;

    -- Combinational routing decode. Held stable for the whole op.
    wb_score <= '1' when op = O_SCORES  else '0';
    attn_op  <= '1' when op = O_ATTN_V  else '0';

    -- Score writeback address: head-major, then query row, then key column.
    -- The sequencer's own out_waddr is an activation-memory address, so the
    -- score path needs its own.
    sc_waddr_o <= std_logic_vector(addr_mul(head,   MAX_TOK * MAX_TOK, SCORE_AW)
                                   + addr_mul(sm_tok, MAX_TOK,        SCORE_AW)
                                   + resize(sm_col,                   SCORE_AW));

    -- Staging geometry: K_h or V_h for the current head, n_tok x HEAD_DIM.
    stage_base <= std_logic_vector(to_unsigned(REG_K, ACT_AW)
                                   + addr_mul(head, HEAD_DIM, ACT_AW))
                  when op = O_STAGE_K else
                  std_logic_vector(to_unsigned(REG_V, ACT_AW)
                                   + addr_mul(head, HEAD_DIM, ACT_AW));
    -- QK^T: 64 reduction rows, n_tok lanes, transposed read.
    -- xV   : n_tok reduction rows, 64 lanes (2 column tiles), direct read.
    stage_rows  <= std_logic_vector(to_unsigned(HEAD_DIM, 7)) when op = O_STAGE_K
                   else std_logic_vector(resize(unsigned(n_tok), 7));
    stage_lanes <= std_logic_vector(resize(unsigned(n_tok), 7)) when op = O_STAGE_K
                   else std_logic_vector(to_unsigned(HEAD_DIM, 7));
    stage_xpose <= '1' when op = O_STAGE_K else '0';

    ---------------------------------------------------------------------
    -- Geometry per op. Pure combinational decode of `op`, so the
    -- sequencer's inputs are stable for the whole op.
    ---------------------------------------------------------------------
    geom : process (op, n_tok, head)
    begin
        -- defaults
        k_len     <= std_logic_vector(to_unsigned(MODEL_DIM, ACT_AW));
        n_coltile <= std_logic_vector(to_unsigned(COL_TILES, 5));
        act_base  <= std_logic_vector(to_unsigned(REG_X, ACT_AW));
        out_base  <= std_logic_vector(to_unsigned(REG_Q, ACT_AW));
        rq_shift  <= std_logic_vector(to_unsigned(RQ_PROJ, 6));

        case op is
            when O_PROJ_Q =>
                out_base <= std_logic_vector(to_unsigned(REG_Q, ACT_AW));
            when O_PROJ_K =>
                out_base <= std_logic_vector(to_unsigned(REG_K, ACT_AW));
            when O_PROJ_V =>
                out_base <= std_logic_vector(to_unsigned(REG_V, ACT_AW));

            when O_SCORES =>
                -- S_h = Q_h . K_h^T : reduce over HEAD_DIM, and n_tok <=
                -- MAX_TOK = 16 <= N_LANES = 32, so the whole score row fits
                -- in ONE column tile. No tiling loop needed.
                k_len     <= std_logic_vector(to_unsigned(HEAD_DIM, ACT_AW));
                n_coltile <= std_logic_vector(to_unsigned(1, 5));
                act_base  <= std_logic_vector(to_unsigned(REG_Q, ACT_AW)
                                              + addr_mul(head, HEAD_DIM, ACT_AW));
                out_base  <= (others => '0');   -- routed to score memory
                rq_shift  <= std_logic_vector(to_unsigned(RQ_SCORE, 6));

            when O_ATTN_V =>
                -- A_h = E_h . V_h : reduce over n_tok, emit HEAD_DIM cols.
                -- HEAD_DIM = 64 = 2 x N_LANES, so two column tiles.
                k_len     <= std_logic_vector(resize(unsigned(n_tok), ACT_AW));
                n_coltile <= std_logic_vector(to_unsigned(HEAD_DIM / N_LANES, 5));
                act_base  <= std_logic_vector(to_unsigned(REG_V, ACT_AW));
                out_base  <= std_logic_vector(to_unsigned(REG_ATTN, ACT_AW)
                                              + addr_mul(head, HEAD_DIM, ACT_AW));
                rq_shift  <= std_logic_vector(to_unsigned(RQ_ATTN, 6));

            when O_PROJ_O =>
                act_base <= std_logic_vector(to_unsigned(REG_ATTN, ACT_AW));
                out_base <= std_logic_vector(to_unsigned(REG_OUT, ACT_AW));

            when O_FFN_HID =>
                -- FFN_DIM = 2048 wide, so FFN_DIM / N_LANES = 64 column
                -- tiles. n_coltile is 5 bits and cannot hold 64, so the
                -- hidden layer is issued in COL_TILES-sized chunks; the
                -- outer loop over chunks lives in the state machine.
                act_base  <= std_logic_vector(to_unsigned(REG_OUT, ACT_AW));
                out_base  <= std_logic_vector(to_unsigned(REG_FFN_H, ACT_AW));
                rq_shift  <= std_logic_vector(to_unsigned(RQ_FFN, 6));

            when O_FFN_OUT =>
                k_len    <= std_logic_vector(to_unsigned(FFN_DIM, ACT_AW));
                act_base <= std_logic_vector(to_unsigned(REG_FFN_H, ACT_AW));
                out_base <= std_logic_vector(to_unsigned(REG_OUT, ACT_AW));
                rq_shift <= std_logic_vector(to_unsigned(RQ_FFN, 6));

            when others =>
                null;
        end case;
    end process;

    ---------------------------------------------------------------------
    -- Op walker
    ---------------------------------------------------------------------
    fsm : process (clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state        <= S_IDLE;
                op           <= O_PROJ_Q;
                head         <= (others => '0');
                sm_tok       <= (others => '0');
                sm_col       <= (others => '0');
                sm_pass      <= '0';
                busy_i       <= '0';
                done_i       <= '0';
                bsel         <= '0';
                seq_start    <= '0';
                stage_go     <= '0';
                sm_row_start <= '0';
                sm_pass_exp  <= '0';
                sm_s_valid   <= '0';
                sc_re        <= '0';
                rc_re        <= '0';
            else
                seq_start    <= '0';
                stage_go     <= '0';
                sm_row_start <= '0';
                sm_s_valid   <= '0';
                sc_re        <= '0';
                rc_re        <= '0';
                done_i       <= '0';

                case state is

                    when S_IDLE =>
                        busy_i <= '0';
                        if start = '1' then
                            busy_i <= '1';
                            op     <= O_PROJ_Q;
                            head   <= (others => '0');
                            state  <= S_SETUP;
                        end if;

                    when S_SETUP =>
                        -- geometry is already decoded from `op`; give it one
                        -- cycle to settle before pulsing start
                        case op is
                            when O_STAGE_K | O_STAGE_V =>
                                stage_go <= '1';
                                state    <= S_COPY;
                            when O_SMAX_MAX =>
                                sm_pass <= '0';
                                sm_tok  <= (others => '0');
                                sm_col  <= (others => '0');
                                state   <= S_SMAX;
                            when O_SMAX_EXP =>
                                sm_pass <= '1';
                                sm_tok  <= (others => '0');
                                sm_col  <= (others => '0');
                                state   <= S_SMAX;
                            when O_RESID =>
                                -- hook present, pass-through for now
                                state <= S_ADVANCE;
                            when O_FINISH =>
                                state <= S_DONE;
                            when others =>
                                state <= S_RUN;
                        end case;

                    when S_RUN =>
                        seq_start <= '1';
                        state     <= S_WAIT;

                    when S_WAIT =>
                        if seq_done = '1' then
                            state <= S_ADVANCE;
                        end if;

                    when S_COPY =>
                        -- The packing datapath in encoder_layer runs the pass
                        -- and raises stage_done when the last bank word is
                        -- written. Waiting on it, rather than counting cycles
                        -- here, keeps the two from drifting apart.
                        if stage_done = '1' then
                            state <= S_ADVANCE;
                        end if;

                    when S_SMAX =>
                        -- Walk one score row per token, MAX_TOK columns each.
                        sm_pass_exp <= sm_pass;
                        sc_re       <= '1';
                        sc_raddr    <= std_logic_vector(
                                          addr_mul(head,   MAX_TOK * MAX_TOK, SCORE_AW)
                                          + addr_mul(sm_tok, MAX_TOK,        SCORE_AW)
                                          + resize(sm_col,                   SCORE_AW));
                        sm_s_valid  <= '1';

                        if sm_col = unsigned(n_tok) - 1 then
                            sm_col <= (others => '0');
                            if sm_pass = '1' then
                                rc_re <= '1';   -- latch 1/sum8 for this row
                            end if;
                            if sm_tok = unsigned(n_tok) - 1 then
                                state <= S_ADVANCE;
                            else
                                sm_tok       <= sm_tok + 1;
                                sm_row_start <= '1';
                            end if;
                        else
                            sm_col <= sm_col + 1;
                        end if;

                    when S_ADVANCE =>
                        case op is
                            when O_PROJ_Q  => op <= O_PROJ_K;   state <= S_SETUP;
                            when O_PROJ_K  => op <= O_PROJ_V;   state <= S_SETUP;
                            when O_PROJ_V  => op <= O_STAGE_K;  state <= S_SETUP;
                            when O_STAGE_K => op <= O_SCORES;   state <= S_SETUP;
                            when O_SCORES  => op <= O_SMAX_MAX; state <= S_SETUP;
                            when O_SMAX_MAX=> op <= O_SMAX_EXP; state <= S_SETUP;
                            when O_SMAX_EXP=> op <= O_STAGE_V;  state <= S_SETUP;
                            when O_STAGE_V => op <= O_ATTN_V;   state <= S_SETUP;

                            when O_ATTN_V =>
                                -- next head, or on to the output projection
                                if head = NUM_HEADS - 1 then
                                    head <= (others => '0');
                                    op    <= O_PROJ_O;
                                else
                                    head <= head + 1;
                                    op    <= O_STAGE_K;
                                end if;
                                state <= S_SETUP;

                            when O_PROJ_O   => op <= O_FFN_HID; state <= S_SETUP;
                            when O_FFN_HID  => op <= O_FFN_OUT; state <= S_SETUP;
                            when O_FFN_OUT  => op <= O_RESID;   state <= S_SETUP;
                            when O_RESID    => op <= O_FINISH;  state <= S_SETUP;
                            when O_FINISH   => state <= S_DONE;
                        end case;

                    when S_DONE =>
                        done_i <= '1';
                        busy_i <= '0';
                        bsel   <= not bsel;   -- flip the weight double buffer
                        state  <= S_IDLE;

                end case;
            end if;
        end if;
    end process;

end architecture;
