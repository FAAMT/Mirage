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
        idle, compute_hl_t0, computer_output_64, result
    );

    type bram_attention_type is array(0 to 255) of signed(17 downto 0);
    type bram_hlw_type is array(0 to 128 * 64 - 1) of signed(17 downto 0); -- 128 neurons, 64 element vec, 
    type bram_olw_type is array(0 to 255 * 64 - 1) of signed(17 downto 0);

    -------------------------------------------------------------------
    -- 2) SIGNAL DECLARATIONS
    -------------------------------------------------------------------
    signal state      : state_type := idle;
    signal prev_state : state_type := idle;

    signal bram_attention : bram_attention_type;
    signal fetch_ctr      : integer := 0;

begin

    -- BRAM Fetch Process
    process (clk, rst)
        variable r_mod_3 : integer;
        variable r_div_3 : integer;
    begin
        if rst = '1' then
            -- BRAM Signals
            bram_attention <= (others => (others => '0'));

            -- Control Signals
            fetch_ctr <= 0;

        elsif rising_edge(clk) then
            if fetch_attn = '1' then
                fetch_ctr <= fetch_ctr + 1;

                if fetch_ctr < 21 then
                    for row in 0 to 11 loop
                        r_mod_3 := row mod 3;
                        r_div_3 := row / 3;

                        bram_attention(r_div_3 * 64 + r_mod_3 + 3 * fetch_ctr) <= resize(signed(output_d(32 * row + 31 downto 32 * row)), 18);
                    end loop;
                else
                    for row in 0 to 11 loop
                        if (row = 0) or (row = 3) or (row = 6) or (row = 9) then
                            r_mod_3 := row mod 3;
                            r_div_3 := row / 3;

                            bram_attention(r_div_3 * 64 + r_mod_3 + 3 * fetch_ctr) <= resize(signed(output_d(32 * row + 31 downto 32 * row)), 18);
                        end if;
                    end loop;
                end if;
            end if;
        end if;
    end process;

    -- Matrix Mult Tiling
    process (clk, rst)
        variable r_mod_3 : integer;
        variable r_div_3 : integer;
    begin
        if rst = '1' then
            state <= idle;
        elsif rising_edge(clk) then
            case state is
                when idle =>
                    if start = '1' then
                        state <= computer_hl_t0;
                    end if;

                when computer_hl_t0 =>

                when others =>
                    state <= idle;
            end case;

        end if;
    end process;

end architecture;