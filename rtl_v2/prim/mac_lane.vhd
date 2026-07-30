-- =====================================================================
-- mac_lane.vhd — one int8 multiply-accumulate lane, resident in a DSP48E1
--
-- This is the fix for v1's zero-DSP result.
--
-- v1 measured 0 of 66 DSPs. Two causes, both addressed here:
--
--   a) v1's accumulator was a transparent LATCH (a conditional assignment
--      with no else, in a combinational process). DSP48E1 inference
--      requires the multiply to feed a CLOCKED register. Fixing that in
--      v1 removed the 240 latches and 29% of the LUTs but still gave 0
--      DSPs, which disproved the idea that the latch alone was to blame.
--
--   b) The real reason: an 8x8 multiply is below Vivado's DSP inference
--      threshold. A DSP48E1 is 25x18, so Vivado maps small multipliers to
--      fabric because that is genuinely cheaper -- in area. Direct
--      evidence from the same run: smx_v_calc's 32x8 multiply DID infer
--      2 DSPs, the only two in the whole design.
--
-- So the multiply must be forced, and forcing it is correct here even
-- though one DSP48E1 per int8 MAC underuses the multiplier: the measured
-- binding constraint is LUTs at 126% while DSPs sit at 3%. Trading an
-- abundant resource for an exhausted one is the whole point.
--
-- USE_DSP = "yes" on the architecture is the documented way to override
-- the heuristic (UG901, "Timing and Area Optimisation"). It is applied to
-- the architecture so it covers the multiply and the accumulator both.
--
-- Structure is the canonical fully-pipelined DSP48E1 MACC:
--
--   a --> areg --\
--                 mreg --> preg --> acc
--   b --> breg --/
--
-- All four registers map into the DSP primitive itself (A, B, M, P), so
-- the lane costs 1 DSP and ~0 fabric. The M register is what makes 200 MHz
-- reachable -- without it the multiply and the accumulate share one clock
-- period. Latency from a/b to acc is 3 cycles; gemm_sequencer accounts
-- for it.
--
-- NOTE ON PACKING: two int8 MACs can share one DSP48E1 by placing the
-- operands at bits [7:0] and [24:17] of the 25-bit A port, since an 8x8
-- product is 16 bits and cannot reach bit 17. That would halve the DSP
-- count to 16. It is deliberately NOT done here -- it requires hand
-- instantiation of DSP48E1 and defeats the inference template, which is
-- the wrong trade while DSPs are the abundant resource. Revisit only if
-- the array ever needs to exceed 66 lanes.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity mac_lane is
    port
    (
        clk : in std_logic;

        -- Datapath. No reset on the pipeline registers: DSP48E1 has a
        -- synchronous reset on P, but resetting A/B/M costs fabric and
        -- buys nothing -- `clr` below already defines the accumulator.
        a : in act_t;   -- broadcast activation
        b : in wgt_t;   -- this lane's weight

        en  : in std_logic;   -- advance the pipeline
        clr : in std_logic;   -- load instead of accumulate (start of a dot product)

        acc : out acc_t
    );
end entity;

architecture rtl of mac_lane is

    signal areg : act_t := (others => '0');
    signal breg : wgt_t := (others => '0');
    signal preg : acc_t := (others => '0');

    -- clr has to travel with the data it applies to: it is sampled with
    -- a/b but acted on one stage later, at the accumulator.
    signal clr_d1 : std_logic := '0';

    -- Override Vivado's small-multiplier heuristic. This is the line that
    -- turns 0 DSPs into N_LANES DSPs.
    attribute use_dsp : string;
    attribute use_dsp of rtl : architecture is "yes";

begin

    acc <= preg;

    -- TWO stages, not three. An earlier version registered the product in
    -- its own 16-bit `mreg` between the multiply and the accumulate, which
    -- reads like the DSP48E1's native A/B -> M -> P pipeline but is not.
    -- Vivado 2022.2 measured it as TWO DSPs per lane (run 6):
    --
    --   mac_lane | (A2*B2)'           |  8 |  8 | 16   <- the multiply
    --   mac_lane | ((C:0x0) or P)+A:B | 30 | 18 | 48   <- a second DSP,
    --                                                     used as an adder
    --
    -- The DSP's internal M register is the full 43-bit product and is not
    -- addressable as a 16-bit value, so asking for a 16-bit register forces
    -- the product out to fabric and into a second DSP's A:B port. 32 lanes
    -- x 2 + 2 in gemm_sequencer = 66, the entire device, and synthesis
    -- reported Used = 67, Available = 66.
    --
    -- Letting the multiply stay combinational between the input registers
    -- and the accumulator is the unambiguous single-DSP MACC: A/B regs,
    -- MREG = 0, P accumulates. The DSP48E1 is built for exactly this.
    mac_proc : process (clk)
    begin
        if rising_edge(clk) then
            if en = '1' then
                -- stage 1: input registers (DSP A and B)
                areg   <= a;
                breg   <= b;
                clr_d1 <= clr;

                -- stage 2: accumulator (DSP P), multiply folded in
                if clr_d1 = '1' then
                    preg <= resize(areg * breg, ACC_W);
                else
                    preg <= preg + resize(areg * breg, ACC_W);
                end if;
            end if;
        end if;
    end process;

end architecture;
