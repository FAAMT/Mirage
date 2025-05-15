library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity ffn_tb is
end ffn_tb;

architecture behavioral of ffn_tb is
    -- Constants
    constant DATA_WIDTH : integer := 17;
    constant CLK_PERIOD : time    := 10 ns;

    -- Component Declaration
    component ffn
        generic
        (
            DATA_WIDTH : integer := 17
        );
        port
        (
            clk        : in std_logic;
            rst        : in std_logic;
            start      : in std_logic;
            done       : out std_logic;
            fetch_attn : in std_logic;
            data_in    : in std_logic_vector(16 * DATA_WIDTH - 1 downto 0);
            start_mac  : out std_logic;
            clear_mac  : out std_logic;
            done_mac   : in std_logic;
            input_a    : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
            input_b    : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
            output_d   : in std_logic_vector(32 * TILE_H - 1 downto 0)
        );
    end component;

    -- Testbench Signals
    signal clk        : std_logic := '0';
    signal rst        : std_logic := '1';
    signal start      : std_logic := '0';
    signal done       : std_logic;
    signal fetch_attn : std_logic                                      := '0';
    signal data_in    : std_logic_vector(16 * DATA_WIDTH - 1 downto 0) := (others => '0');
    signal start_mac  : std_logic;
    signal clear_mac  : std_logic;
    signal done_mac   : std_logic := '0';
    signal input_a    : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal input_b    : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal output_d   : std_logic_vector(32 * TILE_H - 1 downto 0) := (others => '0');

    -- Hardcoded attention matrix values
    type attention_matrix is array(0 to 5, 0 to 3) of integer;
    constant ATTENTION_VALUES : attention_matrix := (
        -- First Q0-Q2 Array
        (-6240, -6240, -6045, -6045),
        (-6045, -6240, -6045, -6045),
        (-6045, -6045, -6045, -6240),

        -- Last Q3 Array Array
        (-2883, -372, -2790, -2883),
        (-372, -2790, -2883, -372),
        (-2790, -2976, -384, -2880)
    );

begin
    -- Clock generation
    clk <= not clk after CLK_PERIOD / 2;

    -- Instantiate DUT
    dut : ffn
    generic
    map (
    DATA_WIDTH => DATA_WIDTH
    )
    port map
    (
        clk        => clk,
        rst        => rst,
        start      => start,
        done       => done,
        fetch_attn => fetch_attn,
        data_in    => data_in,
        start_mac  => start_mac,
        clear_mac  => clear_mac,
        done_mac   => done_mac,
        input_a    => input_a,
        input_b    => input_b,
        output_d   => output_d
    );

    -- Stimulus process
    stim_proc : process
        variable r_idx, c_idx : integer := 0;
    begin
        -- Reset
        rst <= '1';
        wait for 100 ns;
        rst <= '0';
        wait for CLK_PERIOD;

        -- Load attention matrix into BRAM
        report "Loading first 12 values of attention matrix into BRAM";
        fetch_attn <= '1';

        -- First load the 3x4 matrix (12 values)
        for row in 0 to 11 loop
            r_idx := row / 4;
            c_idx := row mod 4;

            report integer'image(row) & " " & integer'image(r_idx) & " " & integer'image(c_idx);
            -- Calculate the proper index and range
            -- Assuming TILE_H is at least 4 (adjust if different) 
            output_d(32 * row + 31 downto 32 * row) <=
            std_logic_vector(to_signed(ATTENTION_VALUES(r_idx, c_idx), 32));
        end loop;
        wait for CLK_PERIOD;

        fetch_attn <= '0';
        wait for 5 * CLK_PERIOD;

        -- Load attention matrix into BRAM
        report "Loading last 4 values of attention matrix into BRAM";
        fetch_attn <= '1';

        -- First load the 3x4 matrix (12 values)
        for row in 0 to 11 loop
            r_idx := row / 4;
            c_idx := row mod 4;

            report integer'image(row) & " " & integer'image(r_idx + 3) & " " & integer'image(c_idx);
            -- Calculate the proper index and range
            -- Assuming TILE_H is at least 4 (adjust if different)
            output_d(32 * row + 31 downto 32 * row) <=
            std_logic_vector(to_signed(ATTENTION_VALUES(r_idx + 3, c_idx), 32));
        end loop;
        wait for CLK_PERIOD;

        fetch_attn <= '0';
        wait for CLK_PERIOD;

        -- Start computation
        report "Starting FFN computation";
        start <= '1';
        wait for CLK_PERIOD;
        start <= '0';

        -- Wait for completion
        wait until done = '1';
        report "FFN computation completed";

        -- Add verification steps here to check outputs

        wait;
    end process;
end architecture;