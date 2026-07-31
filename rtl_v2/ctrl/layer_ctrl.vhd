-- =====================================================================
-- layer_ctrl.vhd — job dispatcher for the encoder datapath
--
-- WHAT CHANGED AND WHY
--
-- This file used to hold a fixed op walker: Q,K,V -> per head (stage,
-- QK^T, softmax x2, stage, xV) -> O -> FFN1 -> FFN2, driven entirely from
-- an internal state machine. Three things made that unworkable, and all
-- three are structural rather than local:
--
--  1. THE WEIGHT WINDOW CANNOT HOLD A PROJECTION. weight_window is
--     BANK_DEPTH = 1024 rows of N_LANES bytes = 32 KB. W_Q alone is
--     512 x 512 = 256 KB, and the FFN output layer reduces over
--     FFN_DIM = 2048 rows, which does not fit even for ONE column tile.
--     So every matmul in this design is inherently split into chunks with
--     a weight reload between them, and something outside the fabric has
--     to perform that reload. A walker that assumes the weights are
--     already there cannot be made correct by fixing its sequencing.
--
--  2. THE EXP STREAM WAS PRODUCED IN ONE OP AND CONSUMED IN ANOTHER.
--     O_SMAX_EXP produced softmax_unit's e_data; O_ATTN_V consumed it via
--     the mac_a mux. They are different ops, so during xV the softmax was
--     not running and mac_a held whatever e_data last happened to be. The
--     attention matmul multiplied V by a stale constant.
--
--  3. THE ROW MAXIMUM WAS A SINGLE REGISTER. PASS_MAX walked every row of
--     a head's score block but kept one max_q, so only the last row's
--     maximum survived; PASS_EXP then re-seeded that same register at each
--     row boundary, leaving rows 1..n-1 computing exp against -2^20. Every
--     one of those rows requantised to a flat 127.
--
-- The fix for all three is the same shape: the PS already has to be in the
-- loop for weights, so let it issue ONE JOB AT A TIME and keep the fabric
-- free of any assumption about what comes next. That is also what the
-- Zynq part is for -- README's own architecture note says "an ARM
-- Cortex-A9 on the PS side orchestrates DMA, tiling and token streaming".
-- This module is now that orchestration's execution end.
--
-- THREE JOB TYPES
--
--   JOB_GEMM   run gemm_sequencer over the geometry in the config
--              registers. Weights must already be in the window.
--   JOB_STAGE  run stage_engine to copy an activation operand into the
--              window (the QK^T and xV operands).
--   JOB_SMAX   two softmax passes over ONE score row: find its maximum,
--              then re-read it, requantise exp(s - max) to int8, write
--              the row to REG_EXP, and latch 1/sum8.
--
-- JOB_SMAX solves (2) and (3) together. The maximum belongs to one row and
-- is used by that row's exp pass immediately, so a single max register is
-- correct by construction rather than by luck. The exp row lands in
-- activation memory, so the xV matmul that follows reads it as an ordinary
-- activation -- no cycle-accurate rendezvous between two ops.
--
-- WHAT IS DELIBERATELY NOT HERE
--
-- Residual addition, layer normalisation, the FFN activation function and
-- the projection biases. Those are O(D) per token against the matmuls'
-- O(D^2) -- 0.4% of the arithmetic, measured -- and they are where int8
-- dynamic range hurts most. They run on the PS, in the same pass it
-- already makes over each tensor to requantise it. See TEST_PLAN.html for
-- the split and its cost.
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

        ---------------------------------------------------------------
        -- Job interface. One job per start pulse; done is a single-cycle
        -- pulse. The PS writes the config registers, then pulses start.
        ---------------------------------------------------------------
        job_start : in  std_logic;
        job_op    : in  std_logic_vector(1 downto 0);
        busy      : out std_logic;
        done      : out std_logic;

        -- Config register file write port. Flat and synchronous; an
        -- AXI-Lite shim on the PS side is a direct map.
        cfg_we   : in std_logic;
        cfg_addr : in std_logic_vector(3 downto 0);
        cfg_data : in std_logic_vector(31 downto 0);

        ---------------------------------------------------------------
        -- gemm_sequencer handshake and geometry
        ---------------------------------------------------------------
        seq_start : out std_logic;
        seq_done  : in  std_logic;
        k_len     : out std_logic_vector(ACT_AW - 1 downto 0);
        n_tok_o   : out std_logic_vector(4 downto 0);
        n_coltile : out std_logic_vector(4 downto 0);
        act_base  : out std_logic_vector(ACT_AW - 1 downto 0);
        out_base  : out std_logic_vector(ACT_AW - 1 downto 0);
        rq_shift  : out std_logic_vector(5 downto 0);
        acc_clr   : out std_logic;   -- clear the accumulator at k = 0
        acc_wb    : out std_logic;   -- write back when the reduction ends

        ---------------------------------------------------------------
        -- stage_engine
        ---------------------------------------------------------------
        stage_go     : out std_logic;
        stage_done   : in  std_logic;
        stage_base   : out std_logic_vector(ACT_AW - 1 downto 0);
        stage_rows   : out std_logic_vector(6 downto 0);
        stage_lanes  : out std_logic_vector(6 downto 0);
        stage_stride : out std_logic_vector(ACT_AW - 1 downto 0);
        stage_xpose  : out std_logic;
        stage_rowofs : out std_logic_vector(BANK_AW - 1 downto 0);

        ---------------------------------------------------------------
        -- softmax control and score-memory reads
        ---------------------------------------------------------------
        sm_row_start : out std_logic;
        sm_pass_exp  : out std_logic;
        sm_s_valid   : out std_logic;
        sm_e_valid   : in  std_logic;
        sc_re        : out std_logic;
        sc_raddr     : out std_logic_vector(SCORE_AW - 1 downto 0);
        rc_re        : out std_logic;

        -- exp row writeback into activation memory
        exp_we    : out std_logic;
        exp_waddr : out std_logic_vector(ACT_AW - 1 downto 0);

        ---------------------------------------------------------------
        -- Datapath routing
        ---------------------------------------------------------------
        wb_score   : out std_logic;   -- route gemm writeback to score memory
        sc_waddr_o : out std_logic_vector(SCORE_AW - 1 downto 0);
        use_recip  : out std_logic;   -- scale the writeback by 1/sum8
        -- Programmable requantisation multiplier in Q(RECIP_FRAC), used for
        -- every op that is not the xV matmul.
        --
        -- This used to be the constant 2^RECIP_FRAC -- unity -- which left
        -- rq_shift as the only requantisation control and therefore
        -- restricted every op to a POWER-OF-TWO output scale. Softmax
        -- cannot live with that: exp_rom is indexed as (max - s) >> 4 over
        -- a table of exp(-i/256), so a score LSB means exactly 2^-12 in
        -- natural-log units. If the QK^T output scale is off by a factor f
        -- the hardware computes exp(-z/f) and the attention runs at the
        -- wrong temperature -- a silent accuracy loss with no failing
        -- endpoint anywhere. A programmable 18-bit multiplier hits the
        -- required scale to about 17 bits instead of to the nearest octave.
        --
        -- Costs one register. The multiplier it feeds already existed for
        -- the reciprocal, and this replaces a constant on that same net.
        rq_scale   : out std_logic_vector(RECIP_ROM_DW - 1 downto 0);
        buf_sel    : out std_logic
    );
end entity;

architecture rtl of layer_ctrl is

    constant JOB_GEMM  : std_logic_vector(1 downto 0) := "00";
    constant JOB_STAGE : std_logic_vector(1 downto 0) := "01";
    constant JOB_SMAX  : std_logic_vector(1 downto 0) := "10";

    type state_t is (
        S_IDLE,
        S_GEMM_GO, S_GEMM_WAIT,
        S_STAGE_GO, S_STAGE_WAIT,
        S_SM_MAX, S_SM_MAXDRAIN, S_SM_EXP, S_SM_EXPDRAIN, S_SM_RECIP,
        S_DONE);
    signal state : state_t := S_IDLE;

    ------------------------------------------------------------------
    -- Config registers, written by the PS.
    ------------------------------------------------------------------
    signal r_k_len     : unsigned(ACT_AW - 1 downto 0) := (others => '0');
    signal r_n_tok     : unsigned(4 downto 0)          := (others => '0');
    signal r_n_coltile : unsigned(4 downto 0)          := (others => '0');
    signal r_rq_shift  : unsigned(5 downto 0)          := (others => '0');
    signal r_acc_clr   : std_logic := '1';
    signal r_acc_wb    : std_logic := '1';
    signal r_wb_score  : std_logic := '0';
    signal r_use_recip : std_logic := '0';
    signal r_act_base  : unsigned(ACT_AW - 1 downto 0) := (others => '0');
    signal r_out_base  : unsigned(ACT_AW - 1 downto 0) := (others => '0');

    signal r_st_base   : unsigned(ACT_AW - 1 downto 0) := (others => '0');
    signal r_st_rows   : unsigned(6 downto 0)          := (others => '0');
    signal r_st_lanes  : unsigned(6 downto 0)          := (others => '0');
    signal r_st_stride : unsigned(ACT_AW - 1 downto 0) := (others => '0');
    signal r_st_xpose  : std_logic := '0';
    signal r_st_rowofs : unsigned(BANK_AW - 1 downto 0) := (others => '0');

    -- softmax job: which score row, and how long it is
    signal r_sm_base : unsigned(SCORE_AW - 1 downto 0) := (others => '0');
    signal r_sm_len  : unsigned(4 downto 0)            := (others => '0');
    -- base of the current head's score block, added to the sequencer's own
    -- writeback address during the QK^T op
    signal r_sc_wbase : unsigned(SCORE_AW - 1 downto 0) := (others => '0');
    signal r_bufsel  : std_logic := '0';
    signal r_scale   : unsigned(RECIP_ROM_DW - 1 downto 0)
                       := to_unsigned(2 ** RECIP_FRAC, RECIP_ROM_DW);

    ------------------------------------------------------------------
    -- softmax walkers
    ------------------------------------------------------------------
    signal sm_col  : unsigned(4 downto 0) := (others => '0');
    signal sm_ecnt : unsigned(4 downto 0) := (others => '0');
    signal sm_drn  : unsigned(2 downto 0) := (others => '0');

    signal busy_i : std_logic := '0';
    signal done_i : std_logic := '0';

begin

    busy    <= busy_i;
    done    <= done_i;
    buf_sel <= r_bufsel;

    -- Geometry is a direct read of the config registers, stable for the
    -- whole job because the PS may not write them while busy.
    k_len        <= std_logic_vector(r_k_len);
    n_tok_o      <= std_logic_vector(r_n_tok);
    n_coltile    <= std_logic_vector(r_n_coltile);
    act_base     <= std_logic_vector(r_act_base);
    out_base     <= std_logic_vector(r_out_base);
    rq_shift     <= std_logic_vector(r_rq_shift);
    acc_clr      <= r_acc_clr;
    acc_wb       <= r_acc_wb;
    wb_score     <= r_wb_score;
    use_recip    <= r_use_recip;
    rq_scale     <= std_logic_vector(r_scale);

    stage_base   <= std_logic_vector(r_st_base);
    stage_rows   <= std_logic_vector(r_st_rows);
    stage_lanes  <= std_logic_vector(r_st_lanes);
    stage_stride <= std_logic_vector(r_st_stride);
    stage_xpose  <= r_st_xpose;
    stage_rowofs <= std_logic_vector(r_st_rowofs);

    -- Base of the current head's score block. encoder_layer adds the
    -- sequencer's own writeback offset to it, so there is exactly one
    -- address generator for the score writeback rather than a second one
    -- here shadowing it.
    sc_waddr_o <= std_logic_vector(r_sc_wbase);

    ------------------------------------------------------------------
    cfg : process (clk)
    begin
        if rising_edge(clk) then
            if cfg_we = '1' then
                case cfg_addr is
                    when x"0" =>
                        r_k_len <= unsigned(cfg_data(ACT_AW - 1 downto 0));
                    when x"1" =>
                        r_n_tok     <= unsigned(cfg_data(4 downto 0));
                        r_n_coltile <= unsigned(cfg_data(12 downto 8));
                        r_rq_shift  <= unsigned(cfg_data(21 downto 16));
                        r_acc_clr   <= cfg_data(24);
                        r_acc_wb    <= cfg_data(25);
                        r_wb_score  <= cfg_data(26);
                        r_use_recip <= cfg_data(27);
                    when x"2" =>
                        r_act_base <= unsigned(cfg_data(ACT_AW - 1 downto 0));
                    when x"3" =>
                        r_out_base <= unsigned(cfg_data(ACT_AW - 1 downto 0));
                    when x"4" =>
                        r_st_base <= unsigned(cfg_data(ACT_AW - 1 downto 0));
                    when x"5" =>
                        r_st_rows   <= unsigned(cfg_data(6 downto 0));
                        r_st_lanes  <= unsigned(cfg_data(14 downto 8));
                        r_st_xpose  <= cfg_data(16);
                        r_st_rowofs <= unsigned(cfg_data(27 downto 18));
                    when x"6" =>
                        r_st_stride <= unsigned(cfg_data(ACT_AW - 1 downto 0));
                    when x"7" =>
                        r_sm_base <= unsigned(cfg_data(SCORE_AW - 1 downto 0));
                        r_sm_len  <= unsigned(cfg_data(20 downto 16));
                    when x"8" =>
                        r_bufsel <= cfg_data(0);
                    when x"9" =>
                        r_sc_wbase <= unsigned(cfg_data(SCORE_AW - 1 downto 0));
                    when x"A" =>
                        r_scale <= unsigned(cfg_data(RECIP_ROM_DW - 1 downto 0));
                    when others =>
                        null;
                end case;
            end if;
        end if;
    end process;

    ------------------------------------------------------------------
    fsm : process (clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state        <= S_IDLE;
                busy_i       <= '0';
                done_i       <= '0';
                seq_start    <= '0';
                stage_go     <= '0';
                sm_row_start <= '0';
                sm_pass_exp  <= '0';
                sm_s_valid   <= '0';
                sc_re        <= '0';
                rc_re        <= '0';
                exp_we       <= '0';
                exp_waddr    <= (others => '0');
                sm_col       <= (others => '0');
                sm_ecnt      <= (others => '0');
                sm_drn       <= (others => '0');
            else
                -- one-cycle-pulse defaults
                seq_start    <= '0';
                stage_go     <= '0';
                sm_row_start <= '0';
                sc_re        <= '0';
                rc_re        <= '0';
                exp_we       <= '0';
                done_i       <= '0';

                -- s_valid trails the score read by exactly one cycle,
                -- which is score_mem's read latency. Both sc_re and
                -- sc_raddr are registered here, so this single delay lines
                -- the softmax up with the data those addresses return.
                --
                -- The as-received code asserted sc_re and sm_s_valid in the
                -- SAME cycle, so softmax_unit sampled the previous
                -- address's score for every column of every row.
                sm_s_valid <= sc_re;

                case state is

                    when S_IDLE =>
                        busy_i <= '0';
                        if job_start = '1' then
                            busy_i <= '1';
                            case job_op is
                                when JOB_GEMM  => state <= S_GEMM_GO;
                                when JOB_STAGE => state <= S_STAGE_GO;
                                when JOB_SMAX  =>
                                    sm_col       <= (others => '0');
                                    sm_ecnt      <= (others => '0');
                                    sm_drn       <= (others => '0');
                                    sm_pass_exp  <= '0';
                                    -- Seed the row maximum and zero the exp
                                    -- sum. Pulsed ONCE per row, before the
                                    -- max pass only: the exp pass must keep
                                    -- the maximum the max pass found.
                                    sm_row_start <= '1';
                                    state        <= S_SM_MAX;
                                when others =>
                                    state <= S_DONE;
                            end case;
                        end if;

                    ----------------------------------------------------
                    when S_GEMM_GO =>
                        seq_start <= '1';
                        state     <= S_GEMM_WAIT;

                    when S_GEMM_WAIT =>
                        if seq_done = '1' then
                            state <= S_DONE;
                        end if;

                    ----------------------------------------------------
                    when S_STAGE_GO =>
                        stage_go <= '1';
                        state    <= S_STAGE_WAIT;

                    when S_STAGE_WAIT =>
                        if stage_done = '1' then
                            state <= S_DONE;
                        end if;

                    ----------------------------------------------------
                    -- Softmax pass 1: walk the row, track the maximum.
                    ----------------------------------------------------
                    when S_SM_MAX =>
                        sc_re    <= '1';
                        sc_raddr <= std_logic_vector(r_sm_base
                                                     + resize(sm_col, SCORE_AW));
                        if sm_col = r_sm_len - 1 then
                            sm_col <= (others => '0');
                            sm_drn <= (others => '0');
                            state  <= S_SM_MAXDRAIN;
                        else
                            sm_col <= sm_col + 1;
                        end if;

                    when S_SM_MAXDRAIN =>
                        -- The last score is still in flight: memory read,
                        -- then softmax_unit's front stage. Starting the exp
                        -- pass before max_q has seen it would compare every
                        -- score against a maximum missing its own last
                        -- column -- the same class of bug the as-received
                        -- rc_re handshake had.
                        if sm_drn = 3 then
                            sm_drn      <= (others => '0');
                            sm_pass_exp <= '1';
                            state       <= S_SM_EXP;
                        else
                            sm_drn <= sm_drn + 1;
                        end if;

                    ----------------------------------------------------
                    -- Softmax pass 2: exp -> int8 -> REG_EXP, and sum8.
                    ----------------------------------------------------
                    when S_SM_EXP =>
                        sc_re    <= '1';
                        sc_raddr <= std_logic_vector(r_sm_base
                                                     + resize(sm_col, SCORE_AW));
                        if sm_col = r_sm_len - 1 then
                            sm_col <= (others => '0');
                            sm_drn <= (others => '0');
                            state  <= S_SM_EXPDRAIN;
                        else
                            sm_col <= sm_col + 1;
                        end if;

                    when S_SM_EXPDRAIN =>
                        -- Let the last exp value reach e_data and be
                        -- accumulated into sum8 before reading the
                        -- reciprocal table.
                        if sm_drn = 4 then
                            sm_drn <= (others => '0');
                            rc_re  <= '1';
                            state  <= S_SM_RECIP;
                        else
                            sm_drn <= sm_drn + 1;
                        end if;

                    when S_SM_RECIP =>
                        -- recip_rom is a synchronous read; give it its
                        -- cycle before declaring the job finished, so the
                        -- xV job that follows sees a settled scale.
                        if sm_drn = 2 then
                            sm_pass_exp <= '0';
                            state       <= S_DONE;
                        else
                            sm_drn <= sm_drn + 1;
                        end if;

                    ----------------------------------------------------
                    when S_DONE =>
                        done_i <= '1';
                        busy_i <= '0';
                        state  <= S_IDLE;

                end case;

                -- The exp stream writes itself into REG_EXP wherever it
                -- appears, independent of the state machine's phase. This
                -- runs outside the case so the tail of the stream still
                -- lands during S_SM_EXPDRAIN, EXP_LAT cycles after the last
                -- score address was issued.
                --
                -- The address is registered from the PRE-increment count,
                -- so it names the byte being written rather than the next
                -- one.
                if sm_e_valid = '1' then
                    exp_we    <= '1';
                    exp_waddr <= std_logic_vector(to_unsigned(REG_EXP, ACT_AW)
                                                  + resize(sm_ecnt, ACT_AW));
                    sm_ecnt   <= sm_ecnt + 1;
                end if;

            end if;
        end if;
    end process;

end architecture;
