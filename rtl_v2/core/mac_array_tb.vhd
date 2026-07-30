-- =====================================================================
-- mac_array_tb.vhd — proves mac_array computes the right dot products
--
-- Self-checking. Drives K reduction steps of pseudo-random int8 data,
-- computes the expected result in the testbench with full precision, and
-- compares every one of the N_LANES accumulators.
--
--   ghdl -r --std=08 mac_array_tb
--
-- Exits with a failure assertion on any mismatch, so it is usable as a
-- regression gate, not just something to eyeball.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity mac_array_tb is
end entity;

architecture tb of mac_array_tb is

    constant CLK_P  : time     := 5 ns;   -- 200 MHz target
    constant K_TEST : positive := 64;     -- reduction length for the test

    signal clk     : std_logic := '0';
    signal a       : act_t     := (others => '0');
    signal w_bus   : wgt_bus_t := (others => '0');
    signal en      : std_logic := '0';
    signal clr     : std_logic := '0';
    signal acc_bus : acc_bus_t;

    signal running : boolean := true;

    -- Deterministic pseudo-random int8 in [-128, 127]. A plain LCG, kept as
    -- a process variable because VHDL-2008 requires shared variables to be
    -- protected types. Reproducible across tools and runs.
    procedure next_rnd(variable s : inout integer; variable r : out integer) is
        constant M : integer := 32749;   -- prime, keeps the product in range
    begin
        s := (s * 137 + 701) mod M;
        r := (s mod 256) - 128;
    end procedure;

begin

    clk_gen : process
    begin
        while running loop
            clk <= '0'; wait for CLK_P / 2;
            clk <= '1'; wait for CLK_P / 2;
        end loop;
        wait;
    end process;

    dut : entity work.mac_array
        port map (clk => clk, a => a, w_bus => w_bus, en => en, clr => clr,
                  acc_bus => acc_bus);

    stim : process
        type exp_arr is array (0 to N_LANES - 1) of integer;
        variable expect : exp_arr := (others => 0);
        variable seed   : integer := 12345;
        variable av     : integer;
        variable wv     : integer;
        variable got    : integer;
        variable errors : natural := 0;
    begin
        -- let the pipeline settle
        en <= '0'; clr <= '0';
        wait until rising_edge(clk);
        wait until rising_edge(clk);

        -- drive K_TEST reduction steps
        en <= '1';
        for k in 0 to K_TEST - 1 loop
            next_rnd(seed, av);
            a  <= to_signed(av, ACT_W);

            for j in 0 to N_LANES - 1 loop
                next_rnd(seed, wv);
                w_bus((j + 1) * WGT_W - 1 downto j * WGT_W)
                    <= std_logic_vector(to_signed(wv, WGT_W));
                -- golden model, full precision
                if k = 0 then
                    expect(j) := av * wv;
                else
                    expect(j) := expect(j) + av * wv;
                end if;
            end loop;

            -- clr on the first term loads rather than accumulates
            if k = 0 then
                clr <= '1';
            else
                clr <= '0';
            end if;

            wait until rising_edge(clk);
        end loop;

        -- Flush the lane pipeline. `en` must stay high for the last product
        -- to reach the accumulator (2 more edges: input reg -> M -> P), but
        -- the inputs must be ZEROED first -- otherwise stage 1 keeps
        -- reloading the final a/w and those extra terms accumulate. Driving
        -- a = 0 makes every flush term 0 * w = 0, so the margin is free.
        clr   <= '0';
        a     <= (others => '0');
        w_bus <= (others => '0');
        for f in 0 to 2 loop
            wait until rising_edge(clk);
        end loop;
        en <= '0';
        wait for CLK_P;

        -- check every lane
        for j in 0 to N_LANES - 1 loop
            got := to_integer(signed(acc_bus((j + 1) * ACC_W - 1 downto j * ACC_W)));
            if got /= expect(j) then
                report "lane " & integer'image(j) & ": expected "
                     & integer'image(expect(j)) & " got " & integer'image(got)
                    severity error;
                errors := errors + 1;
            end if;
        end loop;

        if errors = 0 then
            report "PASS: all " & integer'image(N_LANES) & " lanes match over "
                 & integer'image(K_TEST) & " reduction steps" severity note;
        else
            report integer'image(errors) & " lane(s) mismatched" severity failure;
        end if;

        running <= false;
        wait;
    end process;

end architecture;
