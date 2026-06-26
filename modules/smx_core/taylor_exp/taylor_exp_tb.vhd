library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity taylor_exp_tb is
end taylor_exp_tb;

architecture behavioral of taylor_exp_tb is

    -- Inputs
    signal clk     : std_logic     := '0';
    signal rst     : std_logic     := '0';
    signal start   : std_logic     := '0';
    signal data_in : score_io_type := (others => (others => '0'));

    -- Outputs
    signal done       : std_logic;
    signal fetch_attn : std_logic;
    signal data_out   : std_logic_vector(12 * SCORE_BRAM_DATA_W - 1 downto 0);

    -- Clock period definitions
    constant clk_period : time := 5 ns;

    -- Test data arrays (from BRAM dump)
    type score_array is array (0 to TOK_CONTEXT * NUM_HEADS - 1) of signed(SCORE_BRAM_DATA_W - 1 downto 0);
    constant test_data_q0 : score_array := (
        to_signed(-437, SCORE_BRAM_DATA_W), to_signed(189, SCORE_BRAM_DATA_W), to_signed(842, SCORE_BRAM_DATA_W), to_signed(762, SCORE_BRAM_DATA_W),
        to_signed(-1857, SCORE_BRAM_DATA_W), to_signed(4819, SCORE_BRAM_DATA_W), to_signed(886, SCORE_BRAM_DATA_W), to_signed(-726, SCORE_BRAM_DATA_W),
        to_signed(275, SCORE_BRAM_DATA_W), to_signed(519, SCORE_BRAM_DATA_W), to_signed(-1007, SCORE_BRAM_DATA_W), to_signed(-3568, SCORE_BRAM_DATA_W),
        to_signed(5702, SCORE_BRAM_DATA_W), to_signed(6601, SCORE_BRAM_DATA_W), to_signed(4076, SCORE_BRAM_DATA_W), to_signed(2039, SCORE_BRAM_DATA_W),
        to_signed(-2304, SCORE_BRAM_DATA_W), to_signed(-1127, SCORE_BRAM_DATA_W), to_signed(359, SCORE_BRAM_DATA_W), to_signed(-1882, SCORE_BRAM_DATA_W),
        to_signed(352, SCORE_BRAM_DATA_W), to_signed(873, SCORE_BRAM_DATA_W), to_signed(-459, SCORE_BRAM_DATA_W), to_signed(-1135, SCORE_BRAM_DATA_W),
        to_signed(2928, SCORE_BRAM_DATA_W), to_signed(6872, SCORE_BRAM_DATA_W), to_signed(3049, SCORE_BRAM_DATA_W), to_signed(868, SCORE_BRAM_DATA_W),
        to_signed(944, SCORE_BRAM_DATA_W), to_signed(1716, SCORE_BRAM_DATA_W), to_signed(-157, SCORE_BRAM_DATA_W), to_signed(-2174, SCORE_BRAM_DATA_W)
    );

    constant test_data_q1 : score_array := (
        to_signed(-596, SCORE_BRAM_DATA_W), to_signed(-902, SCORE_BRAM_DATA_W), to_signed(843, SCORE_BRAM_DATA_W), to_signed(1221, SCORE_BRAM_DATA_W),
        to_signed(-393, SCORE_BRAM_DATA_W), to_signed(-1972, SCORE_BRAM_DATA_W), to_signed(687, SCORE_BRAM_DATA_W), to_signed(300, SCORE_BRAM_DATA_W),
        to_signed(3274, SCORE_BRAM_DATA_W), to_signed(9097, SCORE_BRAM_DATA_W), to_signed(4292, SCORE_BRAM_DATA_W), to_signed(2442, SCORE_BRAM_DATA_W),
        to_signed(6240, SCORE_BRAM_DATA_W), to_signed(12915, SCORE_BRAM_DATA_W), to_signed(6089, SCORE_BRAM_DATA_W), to_signed(4631, SCORE_BRAM_DATA_W),
        to_signed(-1040, SCORE_BRAM_DATA_W), to_signed(-3078, SCORE_BRAM_DATA_W), to_signed(-642, SCORE_BRAM_DATA_W), to_signed(-3590, SCORE_BRAM_DATA_W),
        to_signed(2665, SCORE_BRAM_DATA_W), to_signed(1576, SCORE_BRAM_DATA_W), to_signed(1165, SCORE_BRAM_DATA_W), to_signed(-10, SCORE_BRAM_DATA_W),
        to_signed(1678, SCORE_BRAM_DATA_W), to_signed(7233, SCORE_BRAM_DATA_W), to_signed(761, SCORE_BRAM_DATA_W), to_signed(1783, SCORE_BRAM_DATA_W),
        to_signed(-772, SCORE_BRAM_DATA_W), to_signed(-776, SCORE_BRAM_DATA_W), to_signed(498, SCORE_BRAM_DATA_W), to_signed(877, SCORE_BRAM_DATA_W)
    );

    constant test_data_q2 : score_array := (
        to_signed(-380, SCORE_BRAM_DATA_W), to_signed(703, SCORE_BRAM_DATA_W), to_signed(746, SCORE_BRAM_DATA_W), to_signed(1493, SCORE_BRAM_DATA_W),
        to_signed(190, SCORE_BRAM_DATA_W), to_signed(1663, SCORE_BRAM_DATA_W), to_signed(679, SCORE_BRAM_DATA_W), to_signed(3196, SCORE_BRAM_DATA_W),
        to_signed(2321, SCORE_BRAM_DATA_W), to_signed(4705, SCORE_BRAM_DATA_W), to_signed(2820, SCORE_BRAM_DATA_W), to_signed(2285, SCORE_BRAM_DATA_W),
        to_signed(2583, SCORE_BRAM_DATA_W), to_signed(4335, SCORE_BRAM_DATA_W), to_signed(7983, SCORE_BRAM_DATA_W), to_signed(4316, SCORE_BRAM_DATA_W),
        to_signed(-1207, SCORE_BRAM_DATA_W), to_signed(-1235, SCORE_BRAM_DATA_W), to_signed(-1372, SCORE_BRAM_DATA_W), to_signed(-979, SCORE_BRAM_DATA_W),
        to_signed(2661, SCORE_BRAM_DATA_W), to_signed(3430, SCORE_BRAM_DATA_W), to_signed(1217, SCORE_BRAM_DATA_W), to_signed(1593, SCORE_BRAM_DATA_W),
        to_signed(597, SCORE_BRAM_DATA_W), to_signed(2424, SCORE_BRAM_DATA_W), to_signed(974, SCORE_BRAM_DATA_W), to_signed(2901, SCORE_BRAM_DATA_W),
        to_signed(934, SCORE_BRAM_DATA_W), to_signed(2510, SCORE_BRAM_DATA_W), to_signed(1952, SCORE_BRAM_DATA_W), to_signed(85, SCORE_BRAM_DATA_W)
    );

    constant test_data_q3 : score_array := (
        to_signed(-294, SCORE_BRAM_DATA_W), to_signed(-1280, SCORE_BRAM_DATA_W), to_signed(199, SCORE_BRAM_DATA_W), to_signed(195, SCORE_BRAM_DATA_W),
        to_signed(-2471, SCORE_BRAM_DATA_W), to_signed(-1200, SCORE_BRAM_DATA_W), to_signed(520, SCORE_BRAM_DATA_W), to_signed(967, SCORE_BRAM_DATA_W),
        to_signed(-239, SCORE_BRAM_DATA_W), to_signed(898, SCORE_BRAM_DATA_W), to_signed(2150, SCORE_BRAM_DATA_W), to_signed(5737, SCORE_BRAM_DATA_W),
        to_signed(180, SCORE_BRAM_DATA_W), to_signed(3401, SCORE_BRAM_DATA_W), to_signed(4523, SCORE_BRAM_DATA_W), to_signed(10143, SCORE_BRAM_DATA_W),
        to_signed(-1491, SCORE_BRAM_DATA_W), to_signed(-1193, SCORE_BRAM_DATA_W), to_signed(-364, SCORE_BRAM_DATA_W), to_signed(-2436, SCORE_BRAM_DATA_W),
        to_signed(-523, SCORE_BRAM_DATA_W), to_signed(-1143, SCORE_BRAM_DATA_W), to_signed(609, SCORE_BRAM_DATA_W), to_signed(1475, SCORE_BRAM_DATA_W),
        to_signed(-752, SCORE_BRAM_DATA_W), to_signed(2599, SCORE_BRAM_DATA_W), to_signed(1988, SCORE_BRAM_DATA_W), to_signed(5629, SCORE_BRAM_DATA_W),
        to_signed(-676, SCORE_BRAM_DATA_W), to_signed(-1724, SCORE_BRAM_DATA_W), to_signed(1249, SCORE_BRAM_DATA_W), to_signed(275, SCORE_BRAM_DATA_W)
    );

    -- Test control signals
    signal test_index      : integer := 0;
    signal test_phase      : integer := 0;
    signal test_done_count : integer := 0;

begin

    -- Instantiate the Unit Under Test (DUT)
    dut : entity work.taylor_exp
        generic
        map
        (
        -- Width of data elements in bram
        DATA_W => SCORE_BRAM_DATA_W
        )
        port map
        (
            clk        => clk,
            rst        => rst,
            start      => start,
            done       => done,
            fetch_attn => fetch_attn,
            data_in    => data_in,
            data_out   => data_out
        );

    -- Clock process definitions
    clk_process : process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process;

    -- Stimulus process
    stim_proc : process
    begin
        -- Reset the system
        rst <= '1';
        wait for clk_period * 2;
        rst <= '0';
        wait for clk_period;

        -- Test all 4 sets of data (Q_Token 0-3)
        for q in 0 to 3 loop
            -- Apply stimulus for current Q_Token
            test_phase <= q;
            test_index <= 0;

            -- Process all 32 values in groups of 4 (8 sets per Q_Token)
            for i in 0 to 7 loop
                -- Load 4 values into the module
                for j in 0 to 3 loop
                    case q is
                        when 0      => data_in(i * 4 + j) <= test_data_q0(i * 4 + j);
                        when 1      => data_in(i * 4 + j) <= test_data_q1(i * 4 + j);
                        when 2      => data_in(i * 4 + j) <= test_data_q2(i * 4 + j);
                        when 3      => data_in(i * 4 + j) <= test_data_q3(i * 4 + j);
                        when others => null;
                    end case;
                end loop;
            end loop;

            -- Pulse start signal
            start <= '1';
            wait for clk_period;
            start <= '0';

            -- Wait for done signal (expect 4 starts and 4 dones per set)
            wait until done = '1';
            test_done_count <= test_done_count + 1;
            wait for clk_period;

            test_index <= test_index + 1;

            wait for clk_period * 5; -- Small delay between Q_Tokens
        end loop;

        -- Simulation complete
        wait for clk_period * 10;
        report "Simulation completed" severity note;
        wait;
    end process;

    -- Monitor process
    monitor_proc : process (clk)
    begin
        if rising_edge(clk) then
            if done = '1' then
                report "Done received for Q_Token " & integer'image(test_phase) &
                    ", set " & integer'image(test_index) &
                    " at time " & time'image(now);
            end if;
        end if;
    end process;

end behavioral;