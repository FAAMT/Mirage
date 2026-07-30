-- =====================================================================
-- sat8_tb -- equivalence test for mirage_v2_pkg.sat8 against the
-- straightforward clamp it replaced.
--
-- WHY THIS EXISTS
--
-- sat8 used to be two wide comparisons:
--
--     if sh > 127 then 127 elsif sh < -128 then -128 else resize(sh, 8)
--
-- On a 48-bit accumulator each of those is a ~12-deep CARRY4 chain, and
-- Vivado run 6 put them on the worst path in the design (-9.485 ns, 24
-- logic levels, 14 of them CARRY4; all ten worst paths ended at out_data).
--
-- The replacement tests whether the top ACC_W-ACT_W+1 bits are all copies
-- of the sign bit, which is an OR-reduce and an AND-reduce -- about two LUT
-- levels. That is a correct rewrite only if the bit test and the arithmetic
-- clamp agree on EVERY input, including the boundaries where an off-by-one
-- would hide. So this checks the boundaries exhaustively and the wide range
-- by sampling.
--
--   ghdl -a --std=08 rtl_v2/pkg/mirage_v2_pkg.vhd
--   ghdl -a --std=08 rtl_v2/pkg/sat8_tb.vhd
--   ghdl -e --std=08 sat8_tb && ghdl -r --std=08 sat8_tb
-- =====================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.mirage_v2_pkg.all;

entity sat8_tb is
end entity;

architecture tb of sat8_tb is

    -- The reference: exactly what sat8 replaced.
    function sat8_ref(v : acc_t) return act_t is
    begin
        if v > to_signed(127, ACC_W) then
            return to_signed(127, ACT_W);
        elsif v < to_signed(-128, ACC_W) then
            return to_signed(-128, ACT_W);
        else
            return resize(v, ACT_W);
        end if;
    end function;

begin

    check : process
        variable v    : acc_t;
        variable got  : act_t;
        variable want : act_t;
        variable errs : natural := 0;
        variable n    : natural := 0;
        variable seed : integer := 12345;   -- small: seed*137 must not overflow 32-bit
        variable p    : acc_t;

        procedure one(val : acc_t) is
        begin
            got  := sat8(val);
            want := sat8_ref(val);
            n    := n + 1;
            if got /= want then
                -- The input is up to 48 bits, so it cannot go through
                -- integer'image. The case number locates it: the sequence
                -- below is deterministic.
                report "MISMATCH on input #" & integer'image(n)
                     & ": sat8 = " & integer'image(to_integer(got))
                     & ", reference = " & integer'image(to_integer(want))
                    severity error;
                errs := errs + 1;
            end if;
        end procedure;
    begin
        ---------------------------------------------------------------
        -- 1. Exhaustive across the boundaries, well past both clamps.
        ---------------------------------------------------------------
        for i in -1024 to 1023 loop
            one(to_signed(i, ACC_W));
        end loop;

        ---------------------------------------------------------------
        -- 2. The exact clamp edges, stated individually so a failure
        --    names the boundary rather than an index.
        ---------------------------------------------------------------
        one(to_signed(127, ACC_W));    -- largest in range
        one(to_signed(128, ACC_W));    -- first positive overflow
        one(to_signed(-128, ACC_W));   -- smallest in range
        one(to_signed(-129, ACC_W));   -- first negative overflow
        one(to_signed(0, ACC_W));
        one(to_signed(-1, ACC_W));

        ---------------------------------------------------------------
        -- 3. Wide range, including values that only differ from the
        --    in-range cases in the high bits sat8 inspects.
        ---------------------------------------------------------------
        -- 2**e via a shift, not integer exponentiation: e reaches 46 and
        -- VHDL's integer is 32-bit, so `2 ** e` overflows at elaboration.
        for e in 8 to ACC_W - 2 loop
            p := shift_left(to_signed(1, ACC_W), e);
            one(p);
            one(-p);
            one(p + 127);
            one(-p - 128);
        end loop;

        -- extremes
        v := (others => '0'); v(ACC_W - 1) := '1';      -- most negative
        one(v);
        v := (others => '1'); v(ACC_W - 1) := '0';      -- most positive
        one(v);

        ---------------------------------------------------------------
        -- 4. Random 48-bit patterns.
        ---------------------------------------------------------------
        -- Small LCG on purpose: VHDL's integer is 32-bit, so the usual
        -- large multipliers overflow at runtime. Bit 8 of the state is used
        -- rather than bit 0, because an LCG's low bit has a short period.
        for k in 0 to 4000 loop
            for b in 0 to ACC_W - 1 loop
                seed := (seed * 137 + 701) mod 65536;
                if (seed / 256) mod 2 = 0 then
                    v(b) := '0';
                else
                    v(b) := '1';
                end if;
            end loop;
            one(v);
        end loop;

        if errs = 0 then
            report "PASS: sat8 matches the arithmetic clamp on all "
                 & integer'image(n) & " inputs"
                severity note;
        else
            report integer'image(errs) & " mismatch(es) out of "
                 & integer'image(n) severity failure;
        end if;

        wait;
    end process;

end architecture;
