-- =====================================================================
-- exp_idx_tb -- equivalence test for mirage_v2_pkg.exp_index against the
-- integer formulation it replaced.
--
-- WHY THIS EXISTS
--
-- softmax_unit used to compute the exp ROM address like this, with `z` and
-- `idx` declared as `integer`:
--
--     z := to_integer(s_data) - to_integer(max_q);
--     if z > 0 then z := 0; elsif z < EXP_Z_MIN then z := EXP_Z_MIN; end if;
--     idx := (-z) / (2 ** EXP_ROM_SH);
--     if idx > 2**EXP_ROM_AW - 1 then idx := 2**EXP_ROM_AW - 1; end if;
--
-- Every one of those ran at 32 bits because `integer` is 32 bits, and the
-- `/` is a truncating division rather than a shift. Vivado run 8 measured
-- the result: the worst path in the whole design was 17 CARRY4 levels from
-- the score BRAM to rom_addr, -6.226 ns with 62% of the delay in logic.
--
-- exp_index does the same job in signed at the natural width, with the
-- subtract reversed so the negate disappears and one clamp doing the work
-- of three. That is only a valid rewrite if it agrees with the reference
-- on every input, especially at the clamp boundaries and at the point
-- where the shift would otherwise overflow the address.
--
--   ghdl -a --std=08 rtl_v2/pkg/mirage_v2_pkg.vhd
--   ghdl -a --std=08 rtl_v2/pkg/exp_idx_tb.vhd
--   ghdl -e --std=08 exp_idx_tb && ghdl -r --std=08 exp_idx_tb
-- =====================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.mirage_v2_pkg.all;

entity exp_idx_tb is
end entity;

architecture tb of exp_idx_tb is

    -- The reference: exactly the integer code exp_index replaced.
    function exp_index_ref(max_v : score_t; s : score_t) return unsigned is
        variable z   : integer;
        variable idx : integer;
    begin
        z := to_integer(s) - to_integer(max_v);
        if z > 0 then
            z := 0;
        elsif z < EXP_Z_MIN then
            z := EXP_Z_MIN;
        end if;
        idx := (-z) / (2 ** EXP_ROM_SH);
        if idx > 2 ** EXP_ROM_AW - 1 then
            idx := 2 ** EXP_ROM_AW - 1;
        end if;
        return to_unsigned(idx, EXP_ROM_AW);
    end function;

begin

    check : process
        variable errs : natural := 0;
        variable n    : natural := 0;
        variable seed : integer := 24680;
        variable mx   : integer;
        variable dd   : integer;

        procedure one(mv : integer; sv : integer) is
            variable got  : unsigned(EXP_ROM_AW - 1 downto 0);
            variable want : unsigned(EXP_ROM_AW - 1 downto 0);
        begin
            got  := exp_index(to_signed(mv, SCORE_W), to_signed(sv, SCORE_W));
            want := exp_index_ref(to_signed(mv, SCORE_W), to_signed(sv, SCORE_W));
            n    := n + 1;
            if got /= want then
                report "MISMATCH max=" & integer'image(mv)
                     & " s=" & integer'image(sv)
                     & ": exp_index = " & integer'image(to_integer(got))
                     & ", reference = " & integer'image(to_integer(want))
                    severity error;
                errs := errs + 1;
            end if;
        end procedure;
    begin
        -- 1. The whole clamp region and both edges of it, exhaustively.
        --    EXP_Z_MIN is -32768, so this sweeps past the floor and past
        --    the point where (mag >> SH) would exceed EXP_ROM_AW bits.
        for d in 0 to 34000 loop
            one(0, -d);
        end loop;

        -- 2. The named boundaries, individually, so a failure says which.
        one(0, 0);                       -- the argmax itself: index 0
        one(0, -1);
        one(0, -(2 ** EXP_ROM_SH));      -- first index step
        one(0, -(2 ** EXP_ROM_SH) + 1);
        one(0, EXP_Z_MIN + 1);           -- just inside the floor
        one(0, EXP_Z_MIN);               -- exactly the floor
        one(0, EXP_Z_MIN - 1);           -- just past it
        one(0, -MAG_MAX);                -- largest unclamped magnitude
        one(0, -MAG_MAX - 1);            -- first clamped magnitude

        -- 3. Out of contract: s > max. Both must return index 0.
        for d in 1 to 200 loop
            one(0, d);
        end loop;
        one(-1000, 1000);

        -- 4. Non-zero maxima, including near the ends of score_t's range,
        --    where the 22-bit subtract could differ from a 32-bit one.
        for k in 0 to 3000 loop
            seed := (seed * 137 + 701) mod 65536;
            mx := seed - 32768;                       -- -32768 .. 32767
            dd := (seed / 7) mod 40000;
            one(mx, mx - dd);
        end loop;

        -- 5. Extremes of score_t itself.
        one(2 ** (SCORE_W - 1) - 1, -(2 ** (SCORE_W - 1)));  -- widest gap
        one(-(2 ** (SCORE_W - 1)), -(2 ** (SCORE_W - 1)));   -- equal, most negative
        one(2 ** (SCORE_W - 1) - 1, 2 ** (SCORE_W - 1) - 1); -- equal, most positive

        if errs = 0 then
            report "PASS: exp_index matches the integer reference on all "
                 & integer'image(n) & " inputs"
                severity note;
        else
            report integer'image(errs) & " mismatch(es) out of "
                 & integer'image(n) severity failure;
        end if;

        wait;
    end process;

end architecture;
