library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity qkv_proj_calc is
    generic
    (
        DATA_W : integer := MHA_DATA_W;
        TILE_H : integer := TILE_H;
        TILE_W : integer := TILE_W
    );
    port
    (
        -- Clk & Rst
        clk : in std_logic;
        rst : in std_logic;

        -- Ctrl Signals
        start : in std_logic;
        done  : out std_logic;

        -- BRAM Signals
        bram_addr     : in integer range 0 to BASE_OFFSET + TOKEN_DIM - 1;
        bram_data_in  : in std_logic_vector(MHA_DATA_W - 1 downto 0);
        bram_data_out : out std_logic_vector(MHA_DATA_W - 1 downto 0);
        bram_wr       : in std_logic;

        -- Matrix Mult Data Signals
        input_a : out int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        input_b : out int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);

        -- Matrix Mult Ctrl Signals
        start_gemm : out std_logic;
        fetch_gemm : out std_logic;
        done_gemm  : in std_logic
    );
end entity;

architecture behavioral of qkv_proj_calc is

    -- Type Declarations
    type state_type is (idle, compute_q, compute_k, compute_v, mac, tile_incr, result);
    type bram_qkv_type is array (0 to BASE_OFFSET) of std_logic_vector(7 downto 0); -- 48KB for W_Q/K/V
    type bram_token_type is array (0 to TOKEN_DIM) of std_logic_vector(7 downto 0); -- 256B for Token Input

    -- BRAM Storage
    signal bram_qkv        : bram_qkv_type;
    signal bram_token      : bram_token_type;
    signal bram_wr_reg     : std_logic;
    signal bram_data_out_i : std_logic_vector(7 downto 0);

    -- State Signals
    signal state      : state_type := idle;
    signal prev_state : state_type := idle;

    -- Internal Signals
    signal input_a_i     : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1) := (others => (others => (others => '0')));
    signal input_b_i     : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1) := (others => (others => (others => '0')));
    signal base_addr_qkv : integer                                           := 0;
    signal tile_w_pass   : integer                                           := 0; -- Sliding Tile Width Counter
    signal tile_h_pass   : integer                                           := 0; -- Sliding Tile Height Counter

begin
    -- BRAM R/W Process
    process (clk)
    begin
        if rising_edge(clk) then
            if (bram_addr   <= BASE_OFFSET - 1) then
                bram_data_out_i <= bram_qkv(bram_addr);
                if bram_wr = '1' then
                    bram_qkv(bram_addr) <= bram_data_in;
                end if;
            elsif (bram_addr >= BASE_OFFSET) and (bram_addr <= BASE_OFFSET + TOKEN_DIM) then
                bram_data_out_i                                 <= bram_token(bram_addr - BASE_OFFSET);
                if bram_wr = '1' then
                    bram_token(bram_addr - BASE_OFFSET) <= bram_data_in;
                end if;
            end if;
        end if;
    end process;

    -- QKV Dynamic Weight and Input Assignment
    -- Q = base = 0
    -- K = base = MODEL_DIM * HEAD_DIM = 32768
    -- V = base = 2 * MODEL_DIM * HEAD_DIM = 65536
    -- QKV Dynamic Weight and Input Assignment
    process (state, tile_w_pass, tile_h_pass, bram_wr)
    begin
        base_addr_qkv <= 5 * tile_w_pass + 12 * MODEL_DIM * tile_h_pass;

        for i in 0 to TILE_H - 1 loop
            for j in 0 to TILE_W - 1 loop
                if (state = tile_incr) or (tile_h_pass > 5 and i >= 4) then
                    input_a_i(i, j) <= (others => '0');
                    input_b_i(i, j) <= (others => '0');

                elsif (TILE_W * tile_w_pass + j) < MODEL_DIM then
                    input_a_i(i, j) <= signed(bram_token(TILE_W * tile_w_pass + j));

                    if tile_h_pass = 5 and i >= 4 then
                        input_b_i(i, j) <= (others => '0');
                    else
                        if state = compute_q or state = idle then
                            input_b_i(i, j) <= signed(bram_qkv(MODEL_DIM * i + j + 5 * tile_w_pass + 12 * MODEL_DIM * tile_h_pass));
                        elsif state = compute_k then
                            input_b_i(i, j) <= signed(bram_qkv(MODEL_DIM * i + j + (MODEL_DIM * HEAD_DIM) + 5 * tile_w_pass + 12 * MODEL_DIM * tile_h_pass));
                        elsif state = compute_v then
                            input_b_i(i, j) <= signed(bram_qkv(MODEL_DIM * i + j + (2 * MODEL_DIM * HEAD_DIM) + 5 * tile_w_pass + 12 * MODEL_DIM * tile_h_pass));
                        end if;
                    end if;
                else
                    input_a_i(i, j) <= (others => '0');
                    input_b_i(i, j) <= (others => '0');
                end if;
            end loop;
        end loop;
    end process;

    -- FSM 
    process (clk, rst)
    begin
        if rst = '1' then
            state       <= idle;
            tile_w_pass <= 0;
            tile_h_pass <= 0;
            start_gemm  <= '0';
            fetch_gemm  <= '0';
            done        <= '0';
        elsif rising_edge(clk) then
            case state is
                when idle =>
                    done <= '0';
                    if start = '1' then
                        state      <= compute_q;
                        start_gemm <= '1';
                    else
                        start_gemm <= '0';
                    end if;

                when mac =>
                    if done_gemm = '1' then
                        if tile_w_pass = 102 then
                            state      <= tile_incr;
                            start_gemm <= '0';
                        else
                            state      <= prev_state;
                            start_gemm <= '1';

                            tile_w_pass <= tile_w_pass + 1;
                        end if;
                    else
                        state <= mac; -- remain in mac state waiting
                    end if;

                when tile_incr =>
                    tile_w_pass <= 0;
                    start_gemm  <= '1';
                    fetch_gemm  <= '1';
                    state       <= prev_state;

                    if tile_h_pass = 5 then
                        tile_h_pass <= 0;

                        if prev_state = compute_q then
                            state <= compute_k;
                        elsif prev_state = compute_k then
                            state <= compute_v;
                        elsif prev_state = compute_v then
                            start_gemm <= '0';
                            state      <= result;
                        end if;
                    else
                        tile_h_pass <= tile_h_pass + 1;
                    end if;

                when compute_q =>
                    fetch_gemm <= '0';
                    start_gemm <= '0';

                    if tile_h_pass > 5 then
                        tile_h_pass <= 0;
                        state       <= compute_k;
                    else
                        state      <= mac;
                        prev_state <= compute_q;
                    end if;

                    prev_state <= compute_q;

                when compute_k =>
                    start_gemm <= '0';
                    fetch_gemm <= '0';

                    if tile_h_pass > 5 then
                        tile_h_pass <= 0;
                        state       <= compute_v;
                    else
                        state <= mac;
                    end if;

                    prev_state <= compute_k;

                when compute_v =>
                    start_gemm <= '0';
                    fetch_gemm <= '0';

                    if tile_h_pass > 5 then
                        tile_h_pass <= 0;
                        state       <= result;
                    else
                        state <= mac;
                    end if;

                    prev_state <= compute_v;

                when result =>
                    done       <= '1';
                    fetch_gemm <= '0';
                    state      <= idle;

                when others =>
                    state <= idle;
            end case;
        end if;
    end process;

    -- Output Assignments
    input_a       <= input_a_i;
    input_b       <= input_b_i;
    bram_data_out <= bram_data_out_i;

end architecture behavioral;