-- gemm_latency_tb.vhd — DRAFT (outputs/sim): self-checking latency + correctness
-- TB for Mirage gemm_array (12x5 INT8 tile). The upstream gemm_array_tb never
-- pulses `clear`, so `done` latches and per-tile latency is unmeasurable; this
-- TB clears between tiles, counts start->done cycles, and checks the 12 dot
-- products against expected values computed in the TB.
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity gemm_latency_tb is
end entity;

architecture tb of gemm_latency_tb is
    constant CLK_PERIOD : integer := 10; -- ns
    signal clk      : std_logic := '0';
    signal rst      : std_logic := '1';
    signal start    : std_logic := '0';
    signal clear    : std_logic := '0';
    signal done     : std_logic;
    signal input_a  : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal input_b  : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal output_d : std_logic_vector(32 * TILE_H - 1 downto 0);

    -- deterministic stimulus: a(i,j) = i+j-6, b(i,j) = i*2-j  (small ints)
    function aval(i, j : integer) return integer is
    begin
        return i + j - 6;
    end function;
    function bval(i, j : integer) return integer is
    begin
        return i * 2 - j;
    end function;
begin
    clk <= not clk after CLK_PERIOD / 2 * 1 ns;

    dut : entity work.gemm_array
        port map (
            clk      => clk,
            rst      => rst,
            start    => start,
            clear    => clear,
            input_a  => input_a,
            input_b  => input_b,
            output_d => output_d,
            done     => done
        );

    stim : process
        variable latency   : integer;
        variable exp_dot   : integer;
        variable got       : integer;
        variable errors    : integer := 0;
        variable lat_total : integer := 0;
    begin
        -- drive constant operands
        for i in 0 to TILE_H - 1 loop
            for j in 0 to TILE_W - 1 loop
                input_a(i, j) <= to_signed(aval(i, j), MM_IN_DATA_W);
                input_b(i, j) <= to_signed(bval(i, j), MM_IN_DATA_W);
            end loop;
        end loop;

        rst <= '1';
        wait for 3 * CLK_PERIOD * 1 ns;
        wait until rising_edge(clk);
        rst <= '0';
        wait until rising_edge(clk);

        for tile in 0 to 2 loop
            -- one-cycle start pulse, then count cycles until done
            start <= '1';
            wait until rising_edge(clk);
            start <= '0';
            latency := 0;
            while done /= '1' loop
                wait until rising_edge(clk);
                latency := latency + 1;
                if latency > 100 then
                    report "FAIL: done never asserted" severity failure;
                end if;
            end loop;
            lat_total := lat_total + latency;
            report "tile " & integer'image(tile) & ": start->done latency = "
                   & integer'image(latency) & " clk";

            -- gemm_array asserts done one cycle BEFORE accum updates (done is
            -- high during the accumulate edge); result valid 1 clk after done.
            wait until rising_edge(clk);

            -- check the 12 dot products (single accumulate after clear)
            for i in 0 to TILE_H - 1 loop
                exp_dot := 0;
                for j in 0 to TILE_W - 1 loop
                    exp_dot := exp_dot + aval(i, j) * bval(i, j);
                end loop;
                got := to_integer(signed(output_d(32 * i + 31 downto 32 * i)));
                if got /= exp_dot then
                    errors := errors + 1;
                    report "MISMATCH row " & integer'image(i) & ": got "
                           & integer'image(got) & " expected "
                           & integer'image(exp_dot) severity error;
                end if;
            end loop;

            -- clear accumulators + done between tiles (what upstream TB missed)
            wait until rising_edge(clk);
            clear <= '1';
            wait until rising_edge(clk);
            clear <= '0';
            wait until rising_edge(clk);
        end loop;

        if errors = 0 then
            report "PASS: 36/36 dot products correct over 3 tiles; mean latency "
                   & integer'image(lat_total / 3) & " clk/tile" severity note;
        else
            report "FAIL: " & integer'image(errors) & " dot-product mismatches"
                   severity failure;
        end if;
        wait;
    end process;
end architecture;
