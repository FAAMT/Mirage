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
    signal mreg : signed(ACT_W + WGT_W - 1 downto 0) := (others => '0');
    signal preg : acc_t := (others => '0');

    -- clr has to travel with the data it applies to: it is sampled with
    -- a/b but acted on two stages later, at the accumulator.
    signal clr_d1 : std_logic := '0';
    signal clr_d2 : std_logic := '0';

    -- THE ATTRIBUTE GOES ON THE SIGNALS, NOT THE ARCHITECTURE.
    --
    -- `use_dsp of rtl : architecture is "yes"` applies to EVERY arithmetic
    -- operation in the architecture -- so it demanded a DSP for the
    -- accumulator as well as the multiply, and Vivado obliged. Runs 6 and 7
    -- both measured two DSPs per lane:
    --
    --   mac_lane | (A2*B2)'           |  8 |  8 | 16   <- the multiply
    --   mac_lane | ((C:0x0) or P)+A:B | 30 | 18 | 48   <- the accumulator
    --
    -- 32 lanes x 2 + the scale multiply = 67 against a device with 66.
    -- Restructuring the pipeline did not help and could not have: the
    -- attribute, not the coding style, was asking for the second DSP.
    --
    -- So it is now scoped per signal, and the two are pulled apart
    -- deliberately. That is the same trade v2 was built on, inverted:
    -- v1 ran out of LUTs with DSPs idle, so arithmetic moved into DSPs.
    -- Run 6 measured 1,969 LUT (13.7%) and 66 DSP (100%), so the exhausted
    -- resource is now the DSP and the accumulator moves back to fabric.
    -- Cost is 32 accumulators of ACC_W bits: roughly 8 CARRY4 and 32 flops
    -- each, against a device with 14,400 LUT and 28,800 FF spare.
    attribute use_dsp : string;
    attribute use_dsp of mreg : signal is "yes";   -- multiply -> DSP48E1
    attribute use_dsp of preg : signal is "no";    -- accumulate -> fabric

begin

    acc <= preg;

    -- MAC_LAT = 3 stages, and the split is now on a real boundary: the
    -- first two stages are the DSP (A/B input registers, then the
    -- registered product), the third is the fabric accumulator. Registering
    -- the product inside the DSP is free here -- there is no fusion left to
    -- break -- and it keeps the fabric adder in a cycle of its own, which
    -- matters after run 6's -9.485 ns.
    mac_proc : process (clk)
    begin
        if rising_edge(clk) then
            if en = '1' then
                -- stage 1: DSP A and B input registers
                areg   <= a;
                breg   <= b;
                clr_d1 <= clr;

                -- stage 2: DSP product register
                mreg   <= areg * breg;
                clr_d2 <= clr_d1;

                -- stage 3: fabric accumulator
                if clr_d2 = '1' then
                    preg <= resize(mreg, ACC_W);
                else
                    preg <= preg + resize(mreg, ACC_W);
                end if;
            end if;
        end if;
    end process;

end architecture;
