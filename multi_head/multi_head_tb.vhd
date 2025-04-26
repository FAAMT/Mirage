library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

entity multi_head is
    generic (

    );
    port (
        -- CLK & RST
        clk : in std_logic;
        rst : in std_logic;

    );
end entity;

architecture behavioral of matrix_mult is

    -- Type Declarations
    type state_type is (idle, reg_inputs, compute, output_result);

    -- Internal signals

begin

    -- FSM 
    process (clk, rst)
    begin
        if rst = '1' then

        elsif rising_edge(clk) then
            case state is

                when idle   =>
                when others =>
                    state <= idle;

            end case;
        end if;
    end process;

end architecture;