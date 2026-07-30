-- =====================================================================
-- mirage_v2_pkg.vhd — dimensions and types for the Mirage v2 datapath
--
-- v2 keeps v1's logical flow exactly as the MVP plan defines it:
--   qkv projection -> QK^T scores -> exp/softmax -> xV -> FFN -> residual
-- What changes is how the arithmetic and the memories are realised.
--
-- v1's measured failure (Vivado 2022.2, xc7z007sclg400-1):
--   18,182 LUT (126% of device) across 5 of 9 root modules
--   0 BRAM, 0 LUT-as-Memory, 2 of 66 DSP
-- Root cause was one design choice: a 12x5 MAC tile fed by reading weight
-- arrays 60-wide. BRAM has 2 read ports, LUTRAM 16. So every memory became
-- flip-flops plus a combinational mux tree, and every int8 multiply became
-- LUT fabric.
--
-- v2 inverts that. The MAC array is OUTPUT-STATIONARY: lane j owns output
-- element j and accumulates over the reduction dimension serially, so
--
--   * the activation is BROADCAST to all lanes -- 1 byte per cycle
--   * weights are 32 CONSECUTIVE bytes per cycle -- 8 BRAM banks x 32 bits,
--     two ports each, never 60
--   * every multiply is a DSP48E1; the accumulators are deliberately in
--     fabric (runs 6-8: LUT fell to 13.7% while DSP hit 100%, so the
--     scarce resource inverted -- see prim/mac_lane.vhd)
--
-- and there is exactly ONE client of the MAC array (gemm_sequencer), which
-- removes the need for the four-way arbiter v1 lacked.
--
-- Design rules this package exists to enforce, all of them lessons from v1:
--   1. no unconstrained `integer` ports  -- v1 had them; they synthesise to
--      32 bits and hide range errors
--   2. no array-of-array ports          -- v1's 480-bit tile ports made
--      out-of-context sizing meaningless
--   3. no readable `out` ports          -- Vivado rejects them (Synth 8-10557)
--   4. memories only via prim/bram_*    -- one canonical inference template
--   5. no array-wide reset on a memory  -- BRAM contents cannot be reset
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

package mirage_v2_pkg is

    ---------------------------------------------------------------------
    -- Model dimensions (Helsinki-class encoder, int8 quantised)
    ---------------------------------------------------------------------
    constant MODEL_DIM : positive := 512;   -- d_model
    constant NUM_HEADS : positive := 8;
    constant HEAD_DIM  : positive := 64;    -- MODEL_DIM / NUM_HEADS
    constant FFN_DIM   : positive := 2048;  -- d_ff, streamed not resident
    constant MAX_TOK   : positive := 16;    -- context window

    ---------------------------------------------------------------------
    -- Datapath widths
    ---------------------------------------------------------------------
    constant ACT_W   : positive := 8;   -- int8 activations
    constant WGT_W   : positive := 8;   -- int8 weights

    -- Accumulator width. This was 48 -- the DSP48E1 P register -- back when
    -- the accumulator lived inside the DSP. It does not any more (see
    -- prim/mac_lane.vhd), so 48 bits would just be fabric nobody uses.
    --
    -- Worst case is the FFN output layer: 2048 terms of int8 x int8.
    --   2048 * 128 * 128 = 2^25, plus a sign bit = 27 bits.
    -- 32 leaves 5 bits of headroom and is asserted below.
    constant ACC_W   : positive := 32;
    constant SCORE_W : positive := 21;  -- QK^T score, as v1
    constant EXP_W   : positive := 24;  -- exp() result, Q-format
    constant SUM_W   : positive := 32;  -- softmax denominator

    -- Longest reduction any op performs, which is what bounds ACC_W. The
    -- FFN output layer reduces over FFN_DIM; every other op is MODEL_DIM or
    -- shorter. mac_array asserts ACC_W against this.
    constant MAX_REDUCE : positive := FFN_DIM;

    -- mac_lane latency, input to accumulator, in clock edges. Declared ONCE
    -- here because three places have to agree on it: mac_lane's pipeline,
    -- gemm_sequencer's S_FLUSH drain, and mac_array_tb's latency check.
    -- They drifted apart twice when this was a literal in each of them.
    constant MAC_LAT : positive := 3;

    ---------------------------------------------------------------------
    -- MAC array geometry
    --
    -- N_LANES is the output tile width AND the DSP count of the array.
    -- 32 of 66 DSP48E1 leaves headroom for softmax and the reciprocal
    -- unit while staying far under the plan's <90% bar.
    --
    -- W_BANKS x BANK_BYTES must equal N_LANES so that one cycle of BRAM
    -- reads feeds every lane exactly once. 8 banks x 4 bytes = 32.
    -- A RAMB36E1 at 32 bits wide is 1024 deep, which is why BANK_DEPTH is
    -- 1024: one bank is exactly one BRAM primitive, so Vivado cannot
    -- split or merge them into something unintended.
    ---------------------------------------------------------------------
    constant N_LANES    : positive := 32;
    constant W_BANKS    : positive := 8;
    constant BANK_BYTES : positive := 4;              -- int8 per bank word
    constant BANK_W     : positive := BANK_BYTES * WGT_W;  -- 32 bits
    constant BANK_DEPTH : positive := 1024;           -- one RAMB36 at 32b

    -- Reduction steps to cover d_model with one activation per cycle.
    constant K_STEPS : positive := MODEL_DIM;         -- 512

    -- Column tiles needed to sweep a MODEL_DIM-wide output.
    constant COL_TILES : positive := MODEL_DIM / N_LANES;  -- 16

    ---------------------------------------------------------------------
    -- Buffer sizing
    ---------------------------------------------------------------------
    constant ACT_DEPTH   : positive := MAX_TOK * MODEL_DIM;      -- 8192 int8
    constant SCORE_DEPTH : positive := NUM_HEADS * MAX_TOK * MAX_TOK; -- 2048

    ---------------------------------------------------------------------
    -- Address widths. Explicit, so no port is ever an unconstrained
    -- integer (v1 lesson 1).
    ---------------------------------------------------------------------
    constant BANK_AW  : positive := 10;  -- log2(BANK_DEPTH)
    constant ACT_AW   : positive := 16;  -- 64 KB activation memory, 16 RAMB36
    constant SCORE_AW : positive := 11;  -- log2(2048)
    constant DDR_AW   : positive := 24;  -- external weight address space

    ---------------------------------------------------------------------
    -- Activation memory map.
    --
    -- One 64 KB byte-addressed memory holds every live tensor at fixed
    -- bases. The sequencer needs exactly one read port and one write port,
    -- which is precisely what prim/bram_sdp provides -- so activations
    -- cost 16 RAMB36 and no arbitration.
    --
    -- The FFN hidden region is sized for ONE token, not all MAX_TOK. FFN
    -- output for a token reduces over all FFN_DIM hidden values of THAT
    -- token only, so processing token-by-token keeps the region at 2 KB
    -- instead of 32 KB. That single scheduling choice is what makes the
    -- whole design fit.
    ---------------------------------------------------------------------
    constant TOK_STRIDE : positive := MODEL_DIM;              -- 512 B / token
    constant REG_X      : natural  := 16#0000#;  -- layer input
    constant REG_Q      : natural  := 16#2000#;
    constant REG_K      : natural  := 16#4000#;
    constant REG_V      : natural  := 16#6000#;
    constant REG_ATTN   : natural  := 16#8000#;  -- attention output
    constant REG_FFN_H  : natural  := 16#A000#;  -- FFN hidden, one token
    constant REG_OUT    : natural  := 16#B000#;  -- layer output

    ---------------------------------------------------------------------
    -- Types. Vectors of scalars only -- no 2-D array ports (v1 lesson 2).
    ---------------------------------------------------------------------
    subtype act_t   is signed(ACT_W - 1 downto 0);
    subtype wgt_t   is signed(WGT_W - 1 downto 0);
    subtype acc_t   is signed(ACC_W - 1 downto 0);
    subtype score_t is signed(SCORE_W - 1 downto 0);

    -- One cycle of weight fetch: N_LANES int8, flattened to a vector so it
    -- crosses module boundaries as plain std_logic_vector.
    constant WGT_BUS_W : positive := N_LANES * WGT_W;   -- 256 bits
    subtype wgt_bus_t is std_logic_vector(WGT_BUS_W - 1 downto 0);

    -- One lane's accumulator readback, likewise flattened.
    constant ACC_BUS_W : positive := N_LANES * ACC_W;   -- 1536 bits
    subtype acc_bus_t is std_logic_vector(ACC_BUS_W - 1 downto 0);

    ---------------------------------------------------------------------
    -- exp() approximation for softmax.
    --
    -- Parameterised here rather than buried in the RTL, which is the MVP
    -- plan's W3 risk mitigation verbatim: "Parameterize coefficients via
    -- package for future tuning".
    --
    -- Domain: softmax scores are max-subtracted, so z <= 0 always. The
    -- unit clamps z to [EXP_Z_MIN, 0] -- below that, exp(z) is smaller
    -- than one LSB of EXP_ONE and contributes nothing to the sum.
    --
    -- Resolution is set by the table geometry below; see
    -- tools/gen_exp_rom.py for the measured error.
    ---------------------------------------------------------------------
    constant EXP_FRAC  : positive := 12;              -- input Q format
    constant EXP_ONE   : positive := 2 ** EXP_FRAC;   -- 1.0 in Q12
    constant EXP_Z_MIN : integer  := -8 * EXP_ONE;    -- clamp floor, -8.0

    -- exp() is a BRAM lookup table, not a polynomial. v2 tried a degree-2
    -- Horner form first and measured 0.98 ABSOLUTE error over [-8, 0] -- a
    -- parabola turns back upward and cannot track exp. The table is exact
    -- to its resolution, costs one RAMB36 instead of two DSPs, and is
    -- retuned by re-running tools/gen_exp_rom.py rather than editing RTL.
    constant EXP_ROM_AW   : positive := 11;               -- 2048 entries
    constant EXP_ROM_DW   : positive := 18;               -- Q16 result
    constant EXP_ROM_FRAC : positive := 16;               -- output Q format
    constant EXP_ROM_SH   : positive := 4;                -- (-z) >> 4 = index

    -- Largest (max - score) the exp path indexes with. Clamping the
    -- magnitude to 2**(AW+SH)-1 makes (mag >> EXP_ROM_SH) fit EXP_ROM_AW
    -- bits BY CONSTRUCTION, so softmax_unit needs one clamp instead of
    -- three -- and a comparison against 2**k-1 is an OR over the high bits,
    -- not a carry chain. Equals -EXP_Z_MIN - 1, which is asserted below.
    constant MAG_MAX : positive := 2 ** (EXP_ROM_AW + EXP_ROM_SH) - 1;

    -- Softmax normalisation. No divider and no materialised probabilities:
    -- unnormalised exp values feed the MAC array as int8 activations, and
    -- one reciprocal lookup scales the whole output vector at the end,
    -- because every element of a (head, query) output shares one exp-sum.
    --
    -- The reachable index range is bounded below by 127: the argmax element
    -- always has z = 0, so its requantised exp is exactly INT8_MAX. That
    -- bound is what lets a flat table replace a normalise stage.
    constant INT8_MAX      : positive := 127;
    constant RECIP_FRAC    : positive := 16;
    constant RECIP_ROM_AW  : positive := 11;   -- indexed by sum8 directly
    constant RECIP_ROM_DW  : positive := 18;
    constant SUM8_MAX      : positive := MAX_TOK * INT8_MAX;   -- 2032
    constant SUM8_W        : positive := 11;

    ---------------------------------------------------------------------
    -- Which matmul the sequencer is currently walking. Same chain as v1,
    -- but executed on one shared MAC array instead of four private ones.
    ---------------------------------------------------------------------
    type gemm_op_t is (
        OP_IDLE,
        OP_PROJ_Q,    -- x . W_Q
        OP_PROJ_K,    -- x . W_K
        OP_PROJ_V,    -- x . W_V
        OP_SCORES,    -- Q . K^T
        OP_ATTN_V,    -- softmax(S) . V
        OP_PROJ_O,    -- concat(heads) . W_O
        OP_FFN_HID,   -- h . W_1
        OP_FFN_OUT    -- a . W_2
    );

    ---------------------------------------------------------------------
    -- Helpers
    ---------------------------------------------------------------------
    -- Slice lane i's int8 weight out of the packed fetch bus.
    function get_wgt(bus_v : wgt_bus_t; lane : natural) return wgt_t;

    -- Saturating narrow of a 48-bit accumulator to int8, with an
    -- arithmetic right shift for requantisation. Replaces v1's
    -- 60-lane combinational `quantizer`, which synthesised to nothing
    -- because it was a constant shift and therefore pure rewiring.
    function requant(acc : acc_t; shift : natural) return act_t;

    -- exp ROM index for one score against the row max. Contract: s <= max.
    -- A function so it can be tested directly against the reference
    -- formulation -- see rtl_v2/pkg/exp_idx_tb.vhd.
    function exp_index(max_v : score_t; s : score_t) return unsigned;

    -- Bits needed to index n values: ceil(log2(n)). Written as a loop, not
    -- 2**k arithmetic, because VHDL's integer is 32-bit and 2**31 overflows.
    function clog2(n : positive) return natural;

    -- Saturating narrow to int8 with NO shift, for a datapath that has
    -- already shifted. Avoids a wide comparator -- see the body.
    function sat8(v : acc_t) return act_t;

    -- idx * stride, as a w-bit address. ALWAYS use this instead of writing
    -- `idx * stride` directly, because numeric_std's
    --
    --     function "*" (L : UNSIGNED; R : NATURAL) return UNSIGNED
    --       is  return L * TO_UNSIGNED(R, L'LENGTH);
    --
    -- converts the natural to the width of the LEFT operand. Two ways that
    -- bites, and Vivado 2022.2 hit both in layer_ctrl on 2026-07-30:
    --
    --   * `head * HEAD_DIM` with a 4-bit head is TO_UNSIGNED(64, 4) = 0, so
    --     the product is 0 for EVERY head. No error from GHDL or Vivado --
    --     measured, every head addressed head 0's K/V. Silent and total.
    --   * the result is L'LENGTH + R'LENGTH wide, so chained multiplies grow
    --     without bound: `resize(head,11) * MAX_TOK * MAX_TOK` is 44 bits,
    --     which IS a hard error (Synth 8-690, target 11 / source 44).
    --
    -- Widen first, then multiply, then trim. `stride` is a power of two at
    -- every call site, so this is a wired shift, not a multiplier -- it does
    -- not compete with mac_array for DSPs.
    function addr_mul(idx : unsigned; stride : natural; w : positive)
        return unsigned;

end package;

package body mirage_v2_pkg is

    function get_wgt(bus_v : wgt_bus_t; lane : natural) return wgt_t is
    begin
        return signed(bus_v((lane + 1) * WGT_W - 1 downto lane * WGT_W));
    end function;

    function sat8(v : acc_t) return act_t is
        constant TOPW : natural := ACC_W - ACT_W + 1;   -- 41 bits
        variable top  : signed(TOPW - 1 downto 0);
    begin
        -- Saturate WITHOUT a wide comparator. v fits in int8 exactly when
        -- bits ACC_W-1 downto ACT_W-1 are all copies of the sign bit, so
        -- this is an all-zeros / all-ones test -- an OR-reduce and an
        -- AND-reduce, about two LUT levels -- instead of the two 48-bit
        -- carry chains that `> 127` and `< -128` synthesise to.
        --
        -- Run 6's worst path (-9.485 ns, 24 logic levels, 14 of them
        -- CARRY4) ran from a DSP accumulator through this function to
        -- out_data, and every one of the ten worst paths ended here.
        top := v(ACC_W - 1 downto ACT_W - 1);
        if top = 0 or top = -1 then
            -- in range, so the low ACT_W bits are already the two's
            -- complement value (the test above proved the sign bits agree)
            return v(ACT_W - 1 downto 0);
        elsif v(ACC_W - 1) = '0' then
            return to_signed(127, ACT_W);
        else
            return to_signed(-128, ACT_W);
        end if;
    end function;

    function requant(acc : acc_t; shift : natural) return act_t is
    begin
        -- Shift and saturate in one expression. gemm_sequencer does NOT use
        -- this -- it needs the shifted value for out_wide as well, so it
        -- shifts once into its own pipeline register and calls sat8. Kept
        -- for anything that wants the whole operation combinationally.
        return sat8(shift_right(acc, shift));
    end function;

    function exp_index(max_v : score_t; s : score_t) return unsigned is
        variable mag  : signed(SCORE_W downto 0);
        variable idxv : unsigned(EXP_ROM_AW - 1 downto 0);
    begin
        mag := resize(max_v, SCORE_W + 1) - resize(s, SCORE_W + 1);
        if mag(SCORE_W) = '1' then
            idxv := (others => '0');      -- s > max: outside the contract
        elsif mag > MAG_MAX then
            idxv := (others => '1');      -- exp(z) below one LSB
        else
            idxv := resize(shift_right(unsigned(mag), EXP_ROM_SH), EXP_ROM_AW);
        end if;
        return idxv;
    end function;

    function clog2(n : positive) return natural is
        variable r : natural  := 0;
        variable v : positive := 1;
    begin
        while v < n loop
            v := v * 2;
            r := r + 1;
        end loop;
        return r;
    end function;

    function addr_mul(idx : unsigned; stride : natural; w : positive)
        return unsigned is
    begin
        -- resize(idx, w) first, so TO_UNSIGNED(stride, w) holds `stride`
        -- rather than truncating it; the product is then 2w wide and the
        -- outer resize trims it back. Guarded so a future dimension change
        -- fails loudly here instead of silently wrapping.
        assert stride < 2 ** w
            report "addr_mul: stride " & integer'image(stride)
                 & " does not fit in " & integer'image(w) & " bits"
            severity failure;
        return resize(resize(idx, w) * stride, w);
    end function;

end package body;
