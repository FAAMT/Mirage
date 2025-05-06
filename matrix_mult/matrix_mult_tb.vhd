-- Testbench for 12x5 Matrix Multiply Core (Fixed Vectors)
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity matrix_mult_tb is
end matrix_mult_tb;

architecture behavior of matrix_mult_tb is

    constant DATA_WIDTH : integer := 8;
    constant TILE_H     : integer := 12;
    constant TILE_W     : integer := 5;
    constant clk_period : time    := 5 ns;

    signal clk   : std_logic := '0';
    signal rst   : std_logic := '0';
    signal start : std_logic := '0';
    signal clear : std_logic := '0';
    signal done  : std_logic := '0';

    signal input_a  : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal input_b  : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal output_d : std_logic_vector(32 * TILE_H - 1 downto 0);
begin

    -- Instantiate the Unit Under Test (UUT)
    uut : entity work.matrix_mult
        generic
        map(
        DATA_WIDTH => DATA_WIDTH,
        TILE_H     => TILE_H,
        TILE_W     => TILE_W
        )
        port map
        (
            clk      => clk,
            rst      => rst,
            start    => start,
            clear    => clear,
            input_a  => input_a,
            input_b  => input_b,
            output_d => output_d,
            done     => done
        );

    -- Clock generation process
    clk_process : process
    begin
        while now < 3 us loop
            clk <= '0';
            wait for clk_period / 2;
            clk <= '1';
            wait for clk_period / 2;
        end loop;
        wait;
    end process;

    stim_proc : process
        variable rand_val : integer;
        variable a_vals   : signed(7 downto 0);
        variable b_vals   : signed(7 downto 0);
    begin
        -- Reset system
        rst <= '1';
        wait for 20 ns;
        rst <= '0';

        -- Repeat for 52 tiles (5 input elements per tile to cover 256-dim vector)
        for tile_index in 0 to 51 loop

            -- Randomize input vector (5 elements)
            for i in 0 to TILE_H - 1 loop
                for j in 0 to TILE_W - 1 loop
                    rand_val := (j * 2 + tile_index * 7 + 3) mod 128;
                    a_vals   := to_signed(rand_val, 8);
                    input_a(i, j) <= signed(a_vals);
                end loop;
            end loop;

            -- Randomize weight matrix input_b (structured mac_array)
            for row in 0 to TILE_H - 1 loop
                for col in 0 to TILE_W - 1 loop
                    rand_val := ((row * TILE_W + col) * 59 + tile_index * 17 + 13) mod 128;
                    input_b(row, col) <= to_signed(rand_val, 8);
                end loop;
            end loop;

            -- Trigger computation
            start <= '1';
            wait until done = '1';
            start <= '0';

            -- Wait until computation is complete
            wait for clk_period;

            report "Tile completed: " & integer'image(tile_index + 1);
        end loop;

        -- Final done
        report "Test completed after 52 tiles.";
        wait for 2 us; -- Extend time explicitly
        report "End of simulation.";
        wait;
    end process;
end architecture behavior;