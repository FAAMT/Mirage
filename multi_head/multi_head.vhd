library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity multi_head is
    generic
    (
        DATA_WIDTH : integer := 8;
        TILE_H     : integer := 12;
        TILE_W     : integer := 5
    );
    port
    (
        -- Clk & Rst
        clk : in std_logic;
        rst : in std_logic;

        -- Control Signals
        start : in std_logic;
        done  : out std_logic;

        -- BRAM Signals
        bram_addr     : integer range 0 to 49400; -- 49151 + 191 + 255
        bram_data_in  : std_logic_vector(7 downto 0);
        bram_data_out : std_logic_vector(7 downto 0);
        bram_wr       : std_logic;

        -- Matrix Mult Signals
        input_a : out signed_vector(0 to TILE_W - 1);
        input_b : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1)

        start_mac : out std_logic;
        done_mac  : in std_logic;
    );
end entity;

architecture behavioral of matrix_mult is
    -- Type Declarations
    type state_type is (idle, compute_q, compute_k, compute_v, mac, softmax, result);
    type bram_qkv_type is array (0 to 49151) of std_logic_vector(7 downto 0); -- 48KB for W_Q/K/V
    type bram_bias_type is array (0 to 191) of std_logic_vector(7 downto 0); -- 192B for Biases
    type bram_token_type is array (0 to 255) of std_logic_vector(7 downto 0); -- 256B for Token Input

    -- BRAM Signals
    signal bram_qkv : bram_qkv_type;
    signal bram_bias : bram_bias_type;
    signal bram_token : bram_token_type;

    -- State Signals
    signal state : state_type := idle;
    signal prev_state : state_type := idle;

    -- Internal Signals
    signal input_a_i : signed_vector(0 to TILE_W - 1) := (others => '0');
    signal input_b_i : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1) := (others => (others => '0'));
    signal base_addr_qkv : integer;
    signal tile_w_pass : integer range 0 to 51 := 0; -- Sliding Tile Width Counter
    signal tile_h_pass : integer range 0 to 5 := 0; -- Sliding Tile Height Counter
begin
    -- BRAM R/W Process
    process (clk)
    begin
        if rising_edge(clk) then
            if (bram_addr <= 49151) then
                bram_data_out <= bram_qkv(bram_addr);
                if bram_wr = '1' then
                    bram_qkv(bram_addr) <= bram_data_in;
                end if;

            elsif (bram_addr > 49151) and (bram_addr <= 49151 + 191) then
                bram_data_out <= bram_bias(bram_addr - 49152);
                if bram_wr = '1' then
                    bram_bias(bram_addr - 49152) <= bram_data_in;
                end if;

            elsif (bram_addr > 49151 + 191) and (bram_addr <= 49151 + 191 + 255) then
                bram_data_out <= bram_token(bram_addr - 49344);
                if bram_wr = '1' then
                    bram_token(bram_addr - 49344) <= bram_data_in;
                end if;
            end if;
        end if;
    end process;

    -- QKV Dynamic Weight Assignment
    process (tile_w_pass, tile_h_pass)
    begin
        base_addr_qkv <= 5 * tile_w_pass + 12 * 256 * tile_h_pass;

        for i in 0 to TILE_H - 1 loop
            for j in 0 to TILE_W - 1 loop
                input_a_i(j) <= signed(bram_token(TILE_W * tile_w_pass + j));
                input_b_i(i, j) <= signed(bram_qkv(256 * i + j + base_addr_qkv));
            end loop;
        end loop;
    end process;

    -- FSM 
    process (clk, rst)
    begin
        if rst = '1' then
            state <= idle;
            tile_w_pass <= 0;
            tile_h_pass <= 0;
            done <= '0';
        elsif rising_edge(clk) then
            case state is
                when idle =>
                    done <= '0';
                    if start = '1' then
                        state <= compute_q;
                    end if;

                when mac =>
                    start_mac <= '0';

                    if done_mac = '1' then
                        state <= prev_state;
                    end if;

                when compute_q =>
                    if tile_w_pass = 51 then
                        tile_w_pass <= 0;
                        if tile_h_pass = 5 then
                            tile_h_pass <= 0;
                            state <= compute_k;
                        else
                            tile_h_pass <= tile_h_pass + 1;
                        end if;
                    else
                        tile_w_pass <= tile_w_pass + 1;
                    end if;

                    state <= mac;
                    start_mac <= '1';
                    prev_state <= compute_q;

                when compute_k =>
                    -- Same logic for K projection
                    -- (Shift base address by offset for K block)
                    -- TODO: Address K offset
                    state <= compute_v;

                when compute_v =>
                    -- Same logic for V projection
                    -- (Shift base address for V block)
                    -- TODO: Address V offset
                    state <= finish;

                when finish =>
                    done <= '1';
                    state <= idle;

                when others =>
                    state <= idle;
            end case;
        end if;
    end process;

    -- Output Assignments
    input_a <= input_a_i;
    input_b <= input_b_i;

end architecture behavioral;