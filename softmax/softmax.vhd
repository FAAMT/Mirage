library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

entity softmax is
    generic
    (
        DATA_WIDTH : integer := 16
    );
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        start : in std_logic;
        done  : out std_logic;

        fetch_qkv : in std_logic;
        data_in   : in std_logic_vector(12 * DATA_WIDTH - 1 downto 0);
        data_out  : out std_logic_vector(12 * DATA_WIDTH - 1 downto 0)
    );
end entity;

architecture rtl of softmax is
    -- Type Declarations
    type state_type is (idle, fetch, load, compute, result);
    type bram_qkv_type is array (0 to 191) of std_logic_vector(DATA_WIDTH - 1 downto 0); -- 1.53KB for QKV projections
    type input_array is array (0 to 11) of signed(DATA_WIDTH - 1 downto 0);

    -- State Signals
    signal state      : state_type := idle;
    signal prev_state : state_type := idle;

    -- Control Signals
    signal fetch_ctr : integer range 0 to 5 := 0;

    -- BRAM Storage
    signal bram_qkv  : bram_qkv_type;
    signal bram_addr : integer range 0 to 191 := 0;
    signal bram_off  : integer                := DATA_WIDTH;
    signal offset    : integer                := 0;

    -- Max 64 Process Signals
    signal current_max : signed(DATA_WIDTH - 1 downto 0);
    signal index       : integer range 0 to 63 := 0;
    signal processing  : std_logic             := '0';
begin

    -- FSM
    process (clk, rst)
    begin
        if rst = '1' then
            state     <= idle;
            done      <= '0';
            data_out  <= (others => '0');
            bram_addr <= 0;
            offset    <= 0;
            bram_qkv  <= (others => (others => '0'));

            elsif rising_edge(clk) then
            case state is
                when idle =>
                    bram_addr <= 0;

                    if start = '1' then
                        state <= fetch;
                    end if;

                when fetch =>
                    if fetch_qkv = '1' then
                        state      <= load;
                        prev_state <= fetch;
                    end if;

                when load =>
                    for i in 0 to 11 loop
                        -- Determine offset based on bram_addr
                        if bram_addr = 5 and i < 4 then
                            offset <= - 8;
                        elsif bram_addr = 11 and i < 4 then
                            offset <= - 16;
                        end if;

                        -- Assign data if within valid range
                        if (bram_addr = 5 or bram_addr = 11) and i < 4 then
                            bram_qkv(12 * bram_addr + i + offset) <= data_in((bram_off * (i + 1) - 1) downto (bram_off * i));
                        elsif bram_addr = 17 and i < 4 then
                            bram_qkv(12 * bram_addr + i + offset) <= data_in((bram_off * (i + 1) - 1) downto (bram_off * i));
                        elsif bram_addr /= 5 and bram_addr /= 11 and bram_addr /= 17 then
                            bram_qkv(12 * bram_addr + i + offset) <= data_in((bram_off * (i + 1) - 1) downto (bram_off * i));
                        end if;
                    end loop;

                    -- Update state and address outside loop
                    if bram_addr = 17 then
                        state <= compute;
                    else
                        state <= prev_state;
                    end if;
                    bram_addr <= bram_addr + 1;
                when others =>
                    state <= idle;
            end case;
        end if;
    end process;
end architecture;