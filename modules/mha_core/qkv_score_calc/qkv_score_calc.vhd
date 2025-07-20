library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity qkv_score_calc is
    generic
    (
        QKV_DATA_W   : integer := QKV_BRAM_DATA_W;
        SCORE_DATA_W : integer := SCORE_BRAM_DATA_W
    );
    port
    (
        clk   : in std_logic;
        rst   : in std_logic;
        start : in std_logic;
        done  : out std_logic;

        -- QKV BRAM Interface
        fetch_qkv    : in std_logic;
        bram_rd      : in std_logic;
        bram_addr    : in integer range 0 to QKV_BRAM_ELEM - 1;
        bram_data_in : in std_logic_vector(TILE_H * QKV_DATA_W - 1 downto 0);

        -- Matrix Mult Control Interface
        start_gemm : out std_logic;
        clear_gemm : out std_logic;
        done_gemm  : in std_logic;

        -- Matrix Mult Data Interface
        input_a  : out int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        input_b  : out int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        output_d : in std_logic_vector(32 * TILE_H - 1 downto 0)
    );
end entity;

architecture rtl of qkv_score_calc is

    -------------------------------------------------------------------
    -- 1) TYPE DECLARATIONS
    -------------------------------------------------------------------
    type state_type is (
        idle, fetch, load,
        compute_scores
    );
    type bram_scores_type is array(0 to SCORE_DATA_W - 1) of signed(SCORE_DATA_W - 1 downto 0);

    -- Constants
    constant QKV_BRAM_DATA_W : integer := 16;
    constant TILE_H          : integer := 12;

    -------------------------------------------------------------------
    -- 2) SIGNAL DECLARATIONS
    -------------------------------------------------------------------
    signal state      : state_type := idle;
    signal prev_state : state_type := idle;

    signal fetch_ctr : integer range 0 to 5 := 0;
    signal delay_ctr : integer              := 0;

    signal fetch_reg  : std_logic;
    signal start_max  : std_logic;
    signal pause_gemm : std_logic;
    signal q_iter     : integer := 0;
    signal k_iter     : integer := 0;
    signal v_iter     : integer := 0;
    signal pass_iter  : integer := 0;

    signal bram_scores   : bram_scores_type;
    signal bram_data_out : bram_proj_type(0 to TILE_W * TILE_H - 1, 0 to 1);
    signal addr_off_0    : integer := 0;
    signal addr_off_1    : integer := 0;

    -- Add these signals to your architecture
    type addr_array is array(0 to 11, 0 to 4) of integer range 0 to 191;
    signal q_addr_0, k_addr_0 : addr_array; -- For q_int_iter < 4
    signal q_addr_1, k_addr_1 : addr_array; -- For q_int_iter = 4

    -- Quantizer Input declarations 
    signal q_quant_0_in : quantizer_input_array;
    signal k_quant_0_in : quantizer_input_array;

    -- Quantizer Outputdeclarations 
    signal q_quant_0_out : quantizer_output_array;
    signal k_quant_0_out : quantizer_output_array;

begin

    -- Instantiate QKV bram (DUT)
    qkv_proj_bram_inst : entity work.qkv_proj_bram
        generic
        map (
        DATA_W => QKV_BRAM_DATA_W
        )
        port map
        (
            clk           => clk,
            rst           => rst,
            fetch_qkv     => fetch_qkv,
            bram_rd       => bram_rd,
            bram_addr     => bram_addr,
            addr_off_0    => addr_off_0,
            addr_off_1    => addr_off_1,
            bram_data_in  => bram_data_in,
            bram_data_out => bram_data_out
        );

    -- Quantizer instantiations (modified)
    q_quantizer_0 : entity work.quantizer
        generic
        map(
        NUM_QUANTIZERS => 60,
        INPUT_WIDTH    => 16,
        OUTPUT_WIDTH   => 8
        )
        port
        map(
        data_in  => q_quant_0_in,
        data_out => q_quant_0_out
        );

    k_quantizer_0 : entity work.quantizer
        generic
        map(
        NUM_QUANTIZERS => 60,
        INPUT_WIDTH    => 16,
        OUTPUT_WIDTH   => 8
        )
        port
        map(
        data_in  => k_quant_0_in,
        data_out => k_quant_0_out
        );
    -------------------------------------------------------------------
    -- 3) MAIN FSM
    -------------------------------------------------------------------
    process (clk, rst)
        variable k_idx, r_mod_4, r_mod_3 : integer := 0;
    begin
        if rst = '1' then
            -- Control Signals
            state      <= idle;
            done       <= '0';
            delay_ctr  <= 0;
            start_gemm <= '0';
            clear_gemm <= '0';
            pause_gemm <= '0';
            -- Loop Signals
            -- Reset Loop Iters
            q_iter    <= 0;
            k_iter    <= 0;
            v_iter    <= 0;
            pass_iter <= 0;

        elsif rising_edge(clk) then
            case state is

                    -----------------------------------------------------------------
                    -- IDLE: wait for start
                    -----------------------------------------------------------------
                when idle =>
                    -- Reset Loop Iters
                    q_iter    <= 0;
                    k_iter    <= 0;
                    v_iter    <= 0;
                    pass_iter <= 0;

                    -- Reset MAC Controls
                    start_gemm <= '0';
                    clear_gemm <= '0';
                    pause_gemm <= '0';

                    -- Reset Counters
                    delay_ctr <= 0;

                    -- Reset Done Signals
                    done <= '0';

                    if start = '1' then
                        state <= compute_scores;
                    end if;

                    -----------------------------------------------------------------
                    -- COMPUTE_SCORES: issue MAC passes and collect 4×4 results
                    -----------------------------------------------------------------
                when compute_scores =>
                    if q_iter < 2 then
                        if k_iter = 2 then
                            q_iter     <= q_iter + 1;
                            k_iter     <= 0;
                            clear_gemm <= '0';
                        elsif k_iter < 2 then
                            if pass_iter = 1 then
                                if start_gemm = '0' and done_gemm = '1' then
                                    k_iter     <= k_iter + 1;
                                    pass_iter  <= 0;
                                    clear_gemm <= '1';

                                elsif start_gemm = '0' and done_gemm = '0' then
                                    start_gemm <= '1';
                                else
                                    start_gemm <= '0';
                                end if;
                            elsif pass_iter < 1 then
                                if start_gemm = '0' and done_gemm = '1' then
                                    pass_iter <= pass_iter + 1;

                                elsif start_gemm = '0' and done_gemm = '0' then
                                    start_gemm <= '1';
                                else
                                    start_gemm <= '0';
                                end if;
                                clear_gemm <= '0';
                            end if;

                            for row in 0 to 11 loop
                                for col in 0 to 4 loop
                                    k_idx := row * 5 + col;

                                    -- 1st Pass: 60x60 Elements
                                    -- 2nd Pass: 4x4 Elements
                                    -- Score -> Q_iter * K_iter
                                    if pass_iter = 1 then
                                        if row < 1 then
                                            if col < 4 then
                                                input_a(row, col) <= signed(q_quant_0_out(TILE_W * row + col));
                                                input_b(row, col) <= signed(k_quant_0_out(TILE_W * row + col));
                                            else
                                                input_a(row, col) <= (others => '0');
                                                input_b(row, col) <= (others => '0');
                                            end if;
                                        else
                                            input_a(row, col) <= (others => '0');
                                            input_b(row, col) <= (others => '0');
                                        end if;
                                    elsif pass_iter < 1 then
                                        input_a(row, col) <= signed(q_quant_0_out(TILE_W * row + col));
                                        input_b(row, col) <= signed(k_quant_0_out(TILE_W * row + col));
                                    else
                                        input_a(row, col) <= (others => '0');
                                        input_b(row, col) <= (others => '0');
                                    end if;
                                end loop;
                            end loop;
                        end if;
                    end if;

                when others =>
                    state <= idle;
            end case;
        end if;
    end process;

    -- Assign Quantizer Array Data
    quantizer_data_assign : process (bram_data_out)
    begin
        -- Assign first column (60 elements) to quantizer 0 inputs
        for i in 0 to 59 loop
            q_quant_0_in(i) <= bram_data_out(i, 0);
            k_quant_0_in(i) <= bram_data_out(i, 1);
        end loop;
    end process;

    -- Assign Addr Offsets Data
    addr_off_assign : process (state, q_iter, k_iter, v_iter, pass_iter)
    begin
        -- Q & K Proj. Access Pattern
        addr_off_0 <= TILE_W * TILE_H * pass_iter + 192 * q_iter;
        addr_off_1 <= HEAD_DIM + TILE_W * TILE_H * pass_iter + 192 * k_iter;
    end process;

end architecture;