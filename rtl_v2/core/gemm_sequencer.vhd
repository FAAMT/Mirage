-- =====================================================================
-- gemm_sequencer.vhd — the only client of the MAC array
--
-- Walks a tiled matrix multiply:
--
--   for ct in 0 .. n_coltile-1          -- N_LANES output columns per tile
--     for t  in 0 .. n_tok-1            -- one output row per token
--       for k in 0 .. k_len-1           -- reduction, one activation/cycle
--         broadcast act[t][k] to all lanes, fetch weight row k of tile ct
--       flush the lane pipeline, then requantise and write back N_LANES int8
--
-- WHY THIS MODULE RESOLVES TWO OF v1'S OPEN ISSUES OUTRIGHT
--
-- v1 had four modules (qkv_proj_calc, qkv_score_calc, multi_head, ffn) each
-- driving its own MAC interface at one shared gemm_array, with no arbiter
-- -- KNOWN_ISSUES section 6. Two of them (multi_head, qkv_proj_calc) also
-- had no result-return port at all, so the MAC output had nowhere to land
-- -- section 4.
--
-- v2 has exactly ONE sequencer. Every matmul in the encoder -- Q/K/V
-- projection, QK^T, xV, output projection, both FFN layers -- is the same
-- triple loop with different base addresses and lengths, supplied by
-- layer_ctrl. So there is no arbitration to design and no missing
-- writeback path: both problems stop existing rather than getting solved.
--
-- TWO DELIBERATE IMPLEMENTATION CHOICES
--
-- 1. NO MULTIPLIERS IN THE ADDRESS PATH. The obvious way to write the
--    address arithmetic is t_cnt * k_len and ct_cnt * k_len. That puts a
--    real multiplier between a counter and a BRAM address every cycle,
--    which is exactly the sort of thing that eats timing slack. Instead
--    the row and tile bases are RUNNING ACCUMULATORS, advanced by one add
--    when the corresponding counter ticks. Same addresses, no multiply.
--
-- 2. NO DYNAMIC SLICE OF acc_bus. Writing
--       acc_bus((to_integer(i)+1)*ACC_W-1 downto to_integer(i)*ACC_W)
--    is accepted by GHDL but its synthesisability is tool-dependent.
--    acc_bus is unpacked once into a proper array and indexed by signal,
--    which every tool implements as a plain mux.
--
-- LATENCY ALIGNMENT (the easy thing to get wrong here)
--
-- Both the activation buffer and the weight window are synchronous-read
-- BRAM: data arrives one cycle after the address. So mac_en and mac_clr are
-- issued one cycle behind the address generator, via *_d1, to line up with
-- the data they belong to. mac_lane then adds its own MAC_LAT-stage
-- pipeline, which S_FLUSH covers.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity gemm_sequencer is
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        -- op handshake
        start : in  std_logic;
        busy  : out std_logic;
        done  : out std_logic;

        -- geometry of this op, from layer_ctrl
        k_len     : in std_logic_vector(ACT_AW - 1 downto 0);  -- reduction length
        n_tok     : in std_logic_vector(4 downto 0);           -- 1 .. MAX_TOK
        n_coltile : in std_logic_vector(4 downto 0);           -- 1 .. COL_TILES
        act_base  : in std_logic_vector(ACT_AW - 1 downto 0);
        out_base  : in std_logic_vector(ACT_AW - 1 downto 0);
        rq_shift  : in std_logic_vector(5 downto 0);           -- requant shift

        -- Multiplicative scale applied to every accumulator before the
        -- requant shift, in Q(RECIP_FRAC). Unity is 2^RECIP_FRAC.
        --
        -- This is how softmax normalisation happens: for the xV op,
        -- layer_ctrl supplies recip_rom[sum8] here, so the whole output
        -- vector is divided by the exp-sum with ONE multiplier in the
        -- serialised writeback path -- no divider, and no probabilities
        -- ever materialised. Every other op passes unity.
        scale     : in std_logic_vector(RECIP_ROM_DW - 1 downto 0);

        -- activation buffer read port
        act_re    : out std_logic;
        act_raddr : out std_logic_vector(ACT_AW - 1 downto 0);
        act_data  : in  act_t;

        -- weight window read port
        w_re  : out std_logic;
        w_row : out std_logic_vector(BANK_AW - 1 downto 0);
        w_bus : in  wgt_bus_t;

        -- MAC array control and readback
        mac_a   : out act_t;
        mac_w   : out wgt_bus_t;
        mac_en  : out std_logic;
        mac_clr : out std_logic;
        acc_bus : in  acc_bus_t;

        -- Writeback. Two narrowings of the SAME accumulator, both valid on
        -- out_we; the top level routes one of them.
        --   out_data : int8, requantised -- for activation tensors
        --   out_wide : SCORE_W bits      -- for QK^T scores, which must keep
        --              more than 8 bits or softmax loses its dynamic range
        --              before the max subtraction can help it
        out_we    : out std_logic;
        out_waddr : out std_logic_vector(ACT_AW - 1 downto 0);
        out_data  : out act_t;
        out_wide  : out score_t
    );
end entity;

architecture rtl of gemm_sequencer is

    type state_t is (S_IDLE, S_MAC, S_FLUSH, S_WB, S_WB_DRAIN, S_NEXT, S_DONE);
    signal state : state_t := S_IDLE;

    -- loop counters
    signal k_cnt  : unsigned(ACT_AW - 1 downto 0) := (others => '0');
    signal t_cnt  : unsigned(4 downto 0)          := (others => '0');
    signal ct_cnt : unsigned(4 downto 0)          := (others => '0');
    signal fl_cnt : unsigned(2 downto 0)          := (others => '0');
    signal wb_cnt : unsigned(5 downto 0)          := (others => '0');

    -- Running bases, so no multiplier appears in the address path.
    --   act_row  = act_base + t * k_len
    --   w_tile   = ct * k_len
    --   out_row  = out_base + t * (n_coltile * N_LANES)
    --   out_tile = ct * N_LANES
    signal act_row  : unsigned(ACT_AW - 1 downto 0)  := (others => '0');
    signal w_tile   : unsigned(BANK_AW - 1 downto 0) := (others => '0');
    signal out_row  : unsigned(ACT_AW - 1 downto 0)  := (others => '0');
    signal out_tile : unsigned(ACT_AW - 1 downto 0)  := (others => '0');
    -- out_row advances by this each token; computed once per op with a
    -- shift, never a multiply.
    signal out_stride : unsigned(ACT_AW - 1 downto 0) := (others => '0');

    -- one-cycle alignment for the BRAM read latency
    signal en_d1  : std_logic := '0';
    signal clr_d1 : std_logic := '0';

    signal busy_i : std_logic := '0';
    signal done_i : std_logic := '0';

    -- acc_bus unpacked, so the writeback mux is a plain array index.
    type acc_arr_t is array (0 to N_LANES - 1) of acc_t;
    signal accs : acc_arr_t;

    -- Writeback pipeline.
    --
    -- Run 6 measured WNS = -9.485 ns against a 10 ns period with hold met
    -- (WHS = +0.110) -- a pure setup problem, one cone of logic far too
    -- deep. S_WB did all of this between two registers:
    --
    --   accs(wb_cnt) -> 48x19 multiply -> >> RECIP_FRAC -> >> rq_shift
    --                -> saturating narrow -> out_data
    --                                and a SECOND >> rq_shift for out_wide
    --
    -- a multi-DSP multiply followed by two 48-bit variable barrel shifters
    -- and a saturating compare. Now FIVE stages: lane select, multiply,
    -- fixed Q16 shift, the variable requant shift (once, shared by both
    -- outputs), then the saturating narrow. `scale` and `rq_shift` are
    -- constant for a whole op, so only the address and a valid bit have to
    -- travel with the data.
    --
    -- Cost: four extra cycles per 32-lane writeback, against a k_len of up
    -- to 512. Throughput stays one lane per cycle.
    signal wb1_acc  : acc_t                                 := (others => '0');
    signal wb1_addr : unsigned(ACT_AW - 1 downto 0)         := (others => '0');
    signal wb1_v    : std_logic                             := '0';
    signal wb2_prod : signed(ACC_W + RECIP_ROM_DW downto 0) := (others => '0');
    signal wb2_addr : unsigned(ACT_AW - 1 downto 0)         := (others => '0');
    signal wb2_v    : std_logic                             := '0';
    signal wb3_nar  : acc_t                                 := (others => '0');
    signal wb3_addr : unsigned(ACT_AW - 1 downto 0)         := (others => '0');
    signal wb3_v    : std_logic                             := '0';
    signal wb4_sh   : acc_t                                 := (others => '0');
    signal wb4_addr : unsigned(ACT_AW - 1 downto 0)         := (others => '0');
    signal wb4_v    : std_logic                             := '0';
    signal wb_drain : unsigned(1 downto 0)                  := (others => '0');

begin

    busy    <= busy_i;
    done    <= done_i;
    mac_a   <= act_data;
    mac_w   <= w_bus;
    mac_en  <= en_d1;
    mac_clr <= clr_d1;

    unpack : for j in 0 to N_LANES - 1 generate
        accs(j) <= signed(acc_bus((j + 1) * ACC_W - 1 downto j * ACC_W));
    end generate;

    fsm : process (clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state      <= S_IDLE;
                k_cnt      <= (others => '0');
                t_cnt      <= (others => '0');
                ct_cnt     <= (others => '0');
                fl_cnt     <= (others => '0');
                wb_cnt     <= (others => '0');
                act_row    <= (others => '0');
                w_tile     <= (others => '0');
                out_row    <= (others => '0');
                out_tile   <= (others => '0');
                out_stride <= (others => '0');
                en_d1      <= '0';
                clr_d1     <= '0';
                busy_i     <= '0';
                done_i     <= '0';
                act_re     <= '0';
                w_re       <= '0';
                out_we     <= '0';
                out_wide   <= (others => '0');
                -- the writeback pipeline's valid bits, so nothing is
                -- written out of a reset
                wb1_v      <= '0';
                wb2_v      <= '0';
                wb3_v      <= '0';
                wb4_v      <= '0';
                wb_drain   <= (others => '0');
            else
                -- defaults, overridden per state. out_we is NOT defaulted
                -- here -- it is driven only by the writeback pipeline's
                -- valid bit, after the case statement.
                done_i <= '0';
                act_re <= '0';
                w_re   <= '0';
                en_d1  <= '0';
                clr_d1 <= '0';

                case state is

                    when S_IDLE =>
                        busy_i <= '0';
                        if start = '1' then
                            busy_i   <= '1';
                            k_cnt    <= (others => '0');
                            t_cnt    <= (others => '0');
                            ct_cnt   <= (others => '0');
                            act_row  <= unsigned(act_base);
                            w_tile   <= (others => '0');
                            out_row  <= unsigned(out_base);
                            out_tile <= (others => '0');
                            -- n_coltile * N_LANES, N_LANES = 32 = 2^5
                            out_stride <= resize(unsigned(n_coltile) & "00000", ACT_AW);
                            state    <= S_MAC;
                        end if;

                    when S_MAC =>
                        -- Issue this cycle's addresses. The data they return
                        -- next cycle is what en_d1/clr_d1 will validate.
                        act_re    <= '1';
                        act_raddr <= std_logic_vector(act_row + k_cnt);
                        w_re      <= '1';
                        w_row     <= std_logic_vector(w_tile + resize(k_cnt, BANK_AW));

                        -- Validate the PREVIOUS cycle's fetch.
                        en_d1 <= '1';
                        if k_cnt = 0 then
                            clr_d1 <= '1';       -- first term: load, not accumulate
                        end if;

                        if k_cnt = unsigned(k_len) - 1 then
                            k_cnt  <= (others => '0');
                            fl_cnt <= (others => '0');
                            state  <= S_FLUSH;
                        else
                            k_cnt <= k_cnt + 1;
                        end if;

                    when S_FLUSH =>
                        -- Drain. Cycle 0 validates the final BRAM fetch;
                        -- the remaining cycles let it traverse mac_lane's
                        -- MAC_LAT stages. Derived from the package constant
                        -- rather than written as a literal, because this
                        -- count and mac_lane's depth drifted apart twice.
                        -- en must NOT
                        -- stay high past cycle 0 -- mac_a still holds the
                        -- last activation and those terms would be
                        -- accumulated a second time. That exact mistake made
                        -- the first mac_array testbench fail on 31 of 32
                        -- lanes.
                        if fl_cnt = 0 then
                            en_d1 <= '1';
                        end if;
                        if fl_cnt = MAC_LAT + 1 then
                            wb_cnt <= (others => '0');
                            state  <= S_WB;
                        else
                            fl_cnt <= fl_cnt + 1;
                        end if;

                    when S_WB =>
                        -- Stage 1 only: pick the lane and form its address.
                        -- One lane per cycle through a single requantiser --
                        -- N_LANES = 32 cycles against a k_len up to 512, so
                        -- under 7% overhead for 1/32 of the requantiser area.
                        -- The multiply and requantisation happen in the
                        -- pipeline below the case statement.
                        wb1_acc  <= accs(to_integer(wb_cnt));
                        wb1_addr <= out_row + out_tile + resize(wb_cnt, ACT_AW);
                        wb1_v    <= '1';
                        if wb_cnt = N_LANES - 1 then
                            wb_drain <= (others => '0');
                            state    <= S_WB_DRAIN;
                        else
                            wb_cnt <= wb_cnt + 1;
                        end if;

                    when S_WB_DRAIN =>
                        -- Stop issuing and let the last beat traverse
                        -- stages 2 through 5. Leaving early would drop the
                        -- final lane's write.
                        wb1_v <= '0';
                        if wb_drain = 3 then
                            state <= S_NEXT;
                        else
                            wb_drain <= wb_drain + 1;
                        end if;

                    when S_NEXT =>
                        if t_cnt = unsigned(n_tok) - 1 then
                            -- finished all tokens for this column tile
                            t_cnt   <= (others => '0');
                            act_row <= unsigned(act_base);
                            out_row <= unsigned(out_base);
                            if ct_cnt = unsigned(n_coltile) - 1 then
                                state <= S_DONE;
                            else
                                ct_cnt   <= ct_cnt + 1;
                                w_tile   <= w_tile + resize(unsigned(k_len), BANK_AW);
                                out_tile <= out_tile + N_LANES;
                                state    <= S_MAC;
                            end if;
                        else
                            t_cnt   <= t_cnt + 1;
                            act_row <= act_row + unsigned(k_len);
                            out_row <= out_row + out_stride;
                            state   <= S_MAC;
                        end if;

                    when S_DONE =>
                        done_i <= '1';
                        busy_i <= '0';
                        state  <= S_IDLE;

                end case;

                -------------------------------------------------------
                -- Writeback pipeline, stages 2 to 4.
                --
                -- Outside the case so it drains regardless of state, and
                -- AFTER it so these assignments win over the defaults
                -- above: out_we is driven by the pipeline's valid bit and
                -- by nothing else.
                -------------------------------------------------------
                -- stage 2: the scale multiply, alone in its own cycle
                wb2_prod <= wb1_acc * signed('0' & scale);
                wb2_addr <= wb1_addr;
                wb2_v    <= wb1_v;

                -- stage 3: the fixed Q16 shift, which is pure rewiring
                wb3_nar  <= resize(shift_right(wb2_prod, RECIP_FRAC), ACC_W);
                wb3_addr <= wb2_addr;
                wb3_v    <= wb2_v;

                -- stage 4: the variable requantisation shift, done ONCE.
                -- out_data and out_wide both need the same shifted value,
                -- and computing it in each of them built two 48-bit barrel
                -- shifters in the same cycle.
                wb4_sh   <= shift_right(wb3_nar, to_integer(unsigned(rq_shift)));
                wb4_addr <= wb3_addr;
                wb4_v    <= wb3_v;

                -- stage 5: saturating narrow for the activation path, plain
                -- truncate for the score path
                out_data  <= sat8(wb4_sh);
                out_wide  <= resize(wb4_sh, SCORE_W);
                out_waddr <= std_logic_vector(wb4_addr);
                out_we    <= wb4_v;
            end if;
        end if;
    end process;

end architecture;
