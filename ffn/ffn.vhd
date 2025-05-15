library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity ffn is
    generic
    (
        DATA_WIDTH : integer := 17
    );
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        start : in std_logic;
        done  : out std_logic;

        fetch_attn : in std_logic;
        data_in    : in std_logic_vector(16 * DATA_WIDTH - 1 downto 0);

        -- Matrix Mult Control Interface
        start_mac : out std_logic;
        clear_mac : out std_logic;
        done_mac  : in std_logic;

        -- Matrix Mult Data Interface
        input_a  : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        input_b  : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        output_d : in std_logic_vector(32 * TILE_H - 1 downto 0)
    );
end entity;

architecture rtl of ffn is
    -------------------------------------------------------------------
    -- 1) TYPE DECLARATIONS
    -------------------------------------------------------------------
    type state_type is (
        idle, compute_hidden_128, computer_output_64, result
    );

    type bram_attention_type is array(0 to 15) of signed(17 downto 0);

    -------------------------------------------------------------------
    -- 2) SIGNAL DECLARATIONS
    -------------------------------------------------------------------
    signal state      : state_type := idle;
    signal prev_state : state_type := idle;

    signal bram_attention : bram_attention_type;
    signal bram_addr      : integer range 0 to 15 := 0;
    signal bram_wr        : std_logic;

    signal fetch_ctr : integer range 0 to 1 := 0;

begin

    -- BRAM Fetch Process
    process (clk, rst)
    begin
        if rst = '1' then
            -- State Signals
            state      <= idle;
            prev_state <= idle;

            -- BRAM Signals
            bram_addr      <= 0;
            bram_attention <= (others => (others => '0'));

            -- Control Signals
            fetch_ctr <= 0;

            elsif rising_edge(clk) then
            if fetch_attn = '1' then
                if fetch_ctr = 0 then
                    fetch_ctr <= fetch_ctr + 1;

                    for i in 0 to 11 loop
                        bram_attention(i) <= resize(signed(output_d(32 * i + 31 downto 32 * i)), 18);
                    end loop;
                    else
                    for i in 0 to 3 loop
                        fetch_ctr <= 0;

                        bram_attention(i + 12) <=
                        resize(signed(output_d(95 + 96 * i downto 64 + 96 * i)), 18) +
                        resize(signed(output_d(63 + 96 * i downto 32 + 96 * i)), 18) +
                        resize(signed(output_d(31 + 96 * i downto 0 + 96 * i)), 18);
                    end loop;
                end if;
            end if;
        end if;
    end process;
end architecture;