library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use IEEE.STD_LOGIC_TEXTIO.all;
use STD.TEXTIO.all;

entity quantizer_tb is
end quantizer_tb;

architecture Behavioral of quantizer_tb is
    -- Constants
    constant CLK_PERIOD : time := 10 ns;

    -- Test configurations
    constant TEST_INPUT_WIDTH  : positive := 32;
    constant TEST_OUTPUT_WIDTH : positive := 10;
    constant TEST_SHIFT_AMOUNT : natural  := 8;

    -- Signals
    signal clk      : std_logic                                       := '0';
    signal reset    : std_logic                                       := '1';
    signal data_in  : std_logic_vector(TEST_INPUT_WIDTH - 1 downto 0) := (others => '0');
    signal data_out : std_logic_vector(TEST_OUTPUT_WIDTH - 1 downto 0);

    -- Instantiate the DUT
    component quantizer
        generic
        (
            INPUT_WIDTH     : positive;
            OUTPUT_WIDTH    : positive;
            SHIFT_AMOUNT    : natural;
            SHIFT_DIRECTION : string
        );
        port
        (
            clk      : in std_logic;
            reset    : in std_logic;
            data_in  : in std_logic_vector;
            data_out : out std_logic_vector
        );
    end component;

begin
    -- Clock generation
    clk <= not clk after CLK_PERIOD / 2;

    -- Instantiate DUT with right shift configuration
    dut_right_shift : quantizer
    generic
    map (
    INPUT_WIDTH     => TEST_INPUT_WIDTH,
    OUTPUT_WIDTH    => TEST_OUTPUT_WIDTH,
    SHIFT_AMOUNT    => TEST_SHIFT_AMOUNT,
    SHIFT_DIRECTION => "RIGHT"
    )
    port map
    (
        clk      => clk,
        reset    => reset,
        data_in  => data_in,
        data_out => data_out
    );

    -- Stimulus process
    stim_proc : process
        procedure test_case(input_val : integer; expected_val : integer) is
        begin
            data_in <= std_logic_vector(to_signed(input_val, TEST_INPUT_WIDTH));
            wait for CLK_PERIOD;
            assert to_integer(signed(data_out)) = expected_val
            report "Test failed for input " & integer'image(input_val) &
                ". Expected " & integer'image(expected_val) &
                ", got " & integer'image(to_integer(signed(data_out)))
                severity error;
        end procedure;
    begin
        -- Initial reset
        reset <= '1';
        wait for CLK_PERIOD * 2;
        reset <= '0';

        -- Test cases for right shift
        test_case(32767, 127);   -- Max positive case
        test_case(-32768, -128); -- Min negative case
        test_case(1024, 4);      -- Intermediate positive
        test_case(-2048, -8);    -- Intermediate negative
        test_case(255, 0);       -- Small positive (should truncate to 0)
        test_case(-256, -1);     -- Small negative

        -- Add more test cases as needed

        wait for CLK_PERIOD;
        report "Right shift tests completed";

        -- You could add similar tests for left shift configuration here
        wait;
    end process;

end Behavioral;