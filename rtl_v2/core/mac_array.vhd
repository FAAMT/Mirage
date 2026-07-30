-- =====================================================================
-- mac_array.vhd — N_LANES output-stationary int8 MAC lanes
--
-- Computes N_LANES dot products concurrently:
--
--     out[j] = sum over k of  a[k] * W[k][j]        j = 0 .. N_LANES-1
--
-- The activation a[k] is BROADCAST to every lane; only the weights differ
-- per lane. That single choice is what makes v2 implementable:
--
--   * activation bandwidth  = 1 byte/cycle  (v1: 60 bytes/cycle)
--   * weight bandwidth      = N_LANES bytes/cycle, at CONSECUTIVE
--     addresses, so it is 8 BRAM banks x 32 bits with 1 read port each
--   * every accumulator sits in a DSP48E1 P register, not in fabric
--
-- v1's 12x5 tile instead needed 60 weights AND 60 activations per cycle
-- from arbitrary addresses, which no FPGA memory can serve -- hence
-- 0 BRAM, 0 DSP, 126% LUT.
--
-- This array is a pure datapath. It holds no loop counters and no
-- addresses; gemm_sequencer drives `en`/`clr` and supplies a/W. That
-- separation is what lets ONE array serve every matmul in the encoder
-- (projections, scores, xV, FFN) and is why v2 needs no MAC arbiter.
--
-- Latency: 3 cycles from (a,b) to acc, per mac_lane. A dot product of
-- K_STEPS terms completes 3 cycles after its last input.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity mac_array is
    port
    (
        clk : in std_logic;

        -- broadcast activation for reduction step k
        a : in act_t;

        -- N_LANES packed int8 weights for step k, lane j at bits
        -- [(j+1)*8-1 : j*8]
        w_bus : in wgt_bus_t;

        en  : in std_logic;   -- advance all lanes
        clr : in std_logic;   -- first term of a new dot product

        -- N_LANES packed 48-bit accumulators
        acc_bus : out acc_bus_t
    );
end entity;

architecture rtl of mac_array is
begin

    -- The accumulator must not wrap. A product of two ACT_W/WGT_W-bit
    -- signed values needs ACT_W+WGT_W bits, and summing MAX_REDUCE of them
    -- adds clog2(MAX_REDUCE) more. The worst case is the FFN output layer,
    -- reducing over FFN_DIM = 2048 terms.
    --
    -- Checked here because ACC_W stopped being "the DSP48E1 P register
    -- width" when the accumulator moved to fabric -- it is now a number
    -- someone could shrink without realising what bounds it.
    assert ACC_W >= ACT_W + WGT_W + clog2(MAX_REDUCE)
        report "ACC_W = " & integer'image(ACC_W) & " is too narrow: "
             & integer'image(MAX_REDUCE) & " terms of "
             & integer'image(ACT_W) & "x" & integer'image(WGT_W)
             & " need " & integer'image(ACT_W + WGT_W + clog2(MAX_REDUCE))
             & " bits"
        severity failure;

    lanes : for j in 0 to N_LANES - 1 generate
        signal acc_j : acc_t;
    begin
        lane : entity work.mac_lane
            port map (
                clk => clk,
                a   => a,
                b   => get_wgt(w_bus, j),
                en  => en,
                clr => clr,
                acc => acc_j
            );

        acc_bus((j + 1) * ACC_W - 1 downto j * ACC_W) <= std_logic_vector(acc_j);
    end generate;

end architecture;
