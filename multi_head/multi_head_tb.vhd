library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use STD.TEXTIO.all;
use IEEE.STD_LOGIC_TEXTIO.all;
use work.types_pkg.all;

entity multi_head_tb is
end entity;

architecture behavior of multi_head_tb is

    -- Constants
    constant DATA_WIDTH : integer := 8;
    constant TILE_H     : integer := 12;
    constant TILE_W     : integer := 5;
    constant clk_period : time    := 5 ns;

    -- Declarations at top (inside architecture)
    type mac_result_array is array (0 to 15, 0 to 11) of std_logic_vector(31 downto 0); -- 16×12 = 192
    signal mac_results  : mac_result_array := (others => (others => (others => '0')));
    signal result_index : integer          := 0;

    -- DUT Signals
    signal clk   : std_logic := '0';
    signal rst   : std_logic := '0';
    signal start : std_logic := '0';
    signal done  : std_logic;

    signal bram_addr     : integer range 0 to 49599     := 0;
    signal bram_data_in  : std_logic_vector(7 downto 0) := (others => '0');
    signal bram_data_out : std_logic_vector(7 downto 0);
    signal bram_wr       : std_logic := '0';

    signal input_a  : signed_vector(0 to TILE_W - 1);
    signal input_b  : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal output_d : std_logic_vector(32 * TILE_H - 1 downto 0); -- 12x INT32 outputs

    signal start_mac : std_logic;
    signal fetch_mac : std_logic;
    signal done_mac  : std_logic := '0'; -- Simulated handshake

begin

    -- Instantiate UUT (multi_head)
    uut : entity work.multi_head
        generic
        map (
        DATA_WIDTH => DATA_WIDTH,
        TILE_H     => TILE_H,
        TILE_W     => TILE_W
        )
        port map
        (
            clk           => clk,
            rst           => rst,
            start         => start,
            done          => done,
            bram_addr     => bram_addr,
            bram_data_in  => bram_data_in,
            bram_data_out => bram_data_out,
            bram_wr       => bram_wr,
            input_a       => input_a,
            input_b       => input_b,
            start_mac     => start_mac,
            fetch_mac     => fetch_mac,
            done_mac      => done_mac
        );

    -- Instantiate Matrix Multiply Core (MAC Array)
    mac_core : entity work.matrix_mult
        generic
        map (
        DATA_WIDTH => DATA_WIDTH,
        TILE_H     => TILE_H,
        TILE_W     => TILE_W
        )
        port
        map (
        clk      => clk,
        rst      => rst,
        start    => start_mac,
        clear    => fetch_mac,
        input_a  => std_logic_vector(flatten_signed_vector(input_a)),
        input_b  => input_b,
        output_d => output_d, -- output not needed yet for this phase
        done     => done_mac
        );

    -- Clock generation
    clk_process : process
    begin
        loop
            clk <= '0';
            wait for clk_period / 2;
            clk <= '1';
            wait for clk_period / 2;
        end loop;
    end process;

    -- Stimulus process
    stim_proc : process
        variable rand_val    : integer;
        variable line_buffer : line;
        file output_file     : text open write_mode is "q_output.txt";
    begin
        -- Reset
        rst <= '1';
        wait for 20 ns;
        rst <= '0';

        -- Fill Token Vector (BRAM: 49344 to 49599)
        for i in 0 to 255 loop
            rand_val := (i * 2 + 3) mod 128;
            bram_addr    <= 49344 + i;
            bram_data_in <= std_logic_vector(to_signed(rand_val, 8));
            bram_wr      <= '1';
            wait for clk_period;
        end loop;
        bram_wr <= '0';
        wait for clk_period;

        -- Fill Q Weights (BRAM: addresses 0 to 64*256-1 = 16383)
        -- Fill full 64x256 Q weight matrix
        for i in 0 to 16383 loop
            rand_val := ((i mod 256) * 59 + (i / 256) * 37 + 13) mod 256;
            bram_addr    <= i;
            bram_data_in <= std_logic_vector(to_signed(rand_val, 8));
            bram_wr      <= '1';
            wait for clk_period;
        end loop;
        bram_wr <= '0';
        wait for 2 * clk_period;

        -- Fill K Weights
        for i in 0 to 16383 loop
            rand_val := ((i * 3 mod 256) * 59 + (i / 256) * 37 + 13) mod 256;
            bram_addr    <= i + 16383 + 1;
            bram_data_in <= std_logic_vector(to_signed(rand_val, 8));
            bram_wr      <= '1';
            wait for clk_period;
        end loop;
        bram_wr <= '0';
        wait for 2 * clk_period;

        -- Fill K Weights
        for i in 0 to 16383 loop
            rand_val := ((i * 5 mod 256) * 59 + (i / 256) * 37 + 13) mod 256;
            bram_addr    <= i + 16383 * 2 + 2;
            bram_data_in <= std_logic_vector(to_signed(rand_val, 8));
            bram_wr      <= '1';
            wait for clk_period;
        end loop;
        bram_wr <= '0';
        wait for 2 * clk_period;

        -- Start the core
        start <= '1';
        wait for clk_period;
        start <= '0';
        -- Capture 6 fetch_mac outputs
        for pass in 0 to 17 loop -- 18 passes × 12 = 192
            wait until fetch_mac = '1';
            wait for clk_period;

            for i in 0 to 11 loop
                mac_results(result_index mod 6, i) <= output_d((i + 1) * 32 - 1 downto i * 32);
            end loop;
            result_index <= result_index + 1;

            wait for clk_period;

            for i in 0 to 11 loop
                if pass * 12 + i < 1216 then
                    line_buffer := null;
                    write(line_buffer, integer'image(to_integer(signed(mac_results(pass mod 6, i)))));
                    writeline(output_file, line_buffer);
                end if;
            end loop;

        end loop;

        wait for 10 * clk_period; -- small stabilization
        wait;
    end process;

end architecture;