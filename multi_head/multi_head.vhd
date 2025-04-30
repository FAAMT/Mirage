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
        bram_addr     : in integer range 0 to 49599; -- 49151 + 191 + 255
        bram_data_in  : in std_logic_vector(7 downto 0);
        bram_data_out : out std_logic_vector(7 downto 0);
        bram_wr       : in std_logic;

        -- Matrix Mult Signals
        input_a : out signed_vector(0 to TILE_W - 1);
        input_b : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);

        start_mac : out std_logic;
        fetch_mac : out std_logic;
        done_mac  : in std_logic
    );
end entity;

architecture behavioral of multi_head is

    -- Type Declarations
    type state_type is (idle, compute_q, compute_k, compute_v, mac, tile_incr, softmax, result);
    type bram_qkv_type is array (0 to 49151) of std_logic_vector(7 downto 0); -- 48KB for W_Q/K/V
    type bram_bias_type is array (0 to 191) of std_logic_vector(7 downto 0); -- 192B for Biases
    type bram_token_type is array (0 to 255) of std_logic_vector(7 downto 0); -- 256B for Token Input

    -- BRAM Storage
    signal bram_qkv : bram_qkv_type;
    signal bram_bias : bram_bias_type;
    signal bram_token : bram_token_type;
    signal bram_wr_reg : std_logic;

    -- State Signals
    signal state : state_type := idle;
    signal prev_state : state_type := idle;

    -- Internal Signals
    signal input_a_i : signed_vector(0 to TILE_W - 1) := (others => (others => '0'));
    signal input_b_i : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1) := (others => (others => (others => '0')));
    signal base_addr_qkv : integer := 0;
    signal bram_data_out_i : std_logic_vector(7 downto 0);
    signal tile_w_pass : integer := 0; -- Sliding Tile Width Counter
    signal tile_h_pass : integer := 0; -- Sliding Tile Height Counter

begin
    -- BRAM R/W Process
    process (clk)
    begin
        if rising_edge(clk) then
            if (bram_addr <= 49151) then
                bram_data_out_i <= bram_qkv(bram_addr);
                if bram_wr = '1' then
                    bram_qkv(bram_addr) <= bram_data_in;
                end if;
            elsif (bram_addr >= 49152) and (bram_addr <= 49343) then
                bram_data_out_i <= bram_bias(bram_addr - 49152);
                if bram_wr = '1' then
                    bram_bias(bram_addr - 49152) <= bram_data_in;
                end if;
            elsif (bram_addr >= 49344) and (bram_addr <= 49599) then
                bram_data_out_i <= bram_token(bram_addr - 49344);
                if bram_wr = '1' then
                    bram_token(bram_addr - 49344) <= bram_data_in;
                end if;
            end if;
        end if;
    end process;

    -- QKV Dynamic Weight and Input Assignment
    -- Q = base = 0
    -- K = base = 256 * 64 = 16384
    -- V = base = 2 * 256 * 64 = 32768
    -- QKV Dynamic Weight and Input Assignment
    process (state, tile_w_pass, tile_h_pass, bram_wr)
    begin
        base_addr_qkv <= 5 * tile_w_pass + 12 * 256 * tile_h_pass;

        for i in 0 to TILE_H - 1 loop
            for j in 0 to TILE_W - 1 loop
                if (state = tile_incr) or (tile_h_pass > 5 and i >= 4) then
                    input_a_i(j) <= (others => '0');
                    input_b_i(i, j) <= (others => '0');

                elsif (TILE_W * tile_w_pass + j) <= 255 then
                    input_a_i(j) <= signed(bram_token(TILE_W * tile_w_pass + j));

                    if tile_h_pass = 5 and i >= 4 then
                        input_b_i(i, j) <= (others => '0');
                    else
                        if state = compute_q then
                            input_b_i(i, j) <= signed(bram_qkv(256 * i + j + 5 * tile_w_pass + 12 * 256 * tile_h_pass));
                        elsif state = compute_k then
                            input_b_i(i, j) <= signed(bram_qkv(256 * i + j + 16384 + 5 * tile_w_pass + 12 * 256 * tile_h_pass));
                        elsif state = compute_v then
                            input_b_i(i, j) <= signed(bram_qkv(256 * i + j + 32768 + 5 * tile_w_pass + 12 * 256 * tile_h_pass));
                        end if;
                    end if;
                else
                    input_a_i(j) <= (others => '0');
                    input_b_i(i, j) <= (others => '0');
                end if;
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
            start_mac <= '0';
            fetch_mac <= '0';
            done <= '0';
        elsif rising_edge(clk) then
            case state is
                when idle =>
                    done <= '0';
                    if start = '1' then
                        state <= compute_q;
                        start_mac <= '1';
                    end if;

                when mac =>
                    if done_mac = '1' then
                        if tile_w_pass = 51 then
                            state <= tile_incr;
                            fetch_mac <= '1';
                        else
                            state <= prev_state;
                            start_mac <= '1';

                            tile_w_pass <= tile_w_pass + 1;
                        end if;
                    else
                        state <= mac; -- remain in mac state waiting
                    end if;

                when tile_incr =>
                    tile_w_pass <= 0;
                    start_mac <= '1';
                    fetch_mac <= '0';
                    state <= prev_state;

                    if tile_h_pass = 5 then
                        tile_h_pass <= 0;

                        if prev_state = compute_q then
                            state <= compute_k;
                        elsif prev_state = compute_k then
                            state <= compute_v;
                        elsif prev_state = compute_v then
                            state <= result;
                        end if;
                    else
                        tile_h_pass <= tile_h_pass + 1;
                    end if;

                when compute_q =>
                    if tile_h_pass > 5 then
                        tile_h_pass <= 0;
                        state <= compute_k;
                    else
                        state <= mac;
                        start_mac <= '0';
                        prev_state <= compute_q;
                    end if;

                    start_mac <= '0';
                    prev_state <= compute_q;

                when compute_k =>
                    if tile_h_pass > 5 then
                        tile_h_pass <= 0;
                        start_mac <= '1';
                        state <= compute_v;
                    else
                        state <= mac;
                    end if;

                    start_mac <= '0';
                    prev_state <= compute_k;

                when compute_v =>
                    if tile_h_pass > 5 then
                        tile_h_pass <= 0;
                        state <= result;
                    else
                        state <= mac;
                    end if;

                    start_mac <= '0';
                    prev_state <= compute_v;

                when result =>
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
    bram_data_out <= bram_data_out_i;

end architecture behavioral;