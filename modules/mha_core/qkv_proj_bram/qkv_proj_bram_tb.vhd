-- ============================================================================
--  File      : qkv_proj_bram_tb.vhd
--  Author    : Fahad Tajiki
--  Date      : 10/07/2025
--  Purpose   : Testbench for QKV bram, verifying correct storage of Query, Key, 
--              and Value vectors computed by matrix multiplication in a Transformer
--              Encoder Layer. Integrates gemm_array and qkv_proj_calc cores, driving
--              the QKV bram with fetch_qkv signal and validating stored data.
-- ============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use STD.TEXTIO.all;
use IEEE.STD_LOGIC_TEXTIO.all;
use work.types_pkg.all;
-- use work.tb_io_pkg.all;

entity qkv_proj_bram_tb is
end entity;

architecture behavior of qkv_proj_bram_tb is
    -- Array to store matrix multiplication results
    type gemm_result_array is array (0 to 15, 0 to 11) of std_logic_vector(31 downto 0);
    signal gemm_results : gemm_result_array := (others => (others => (others => '0')));

    -- Matrix type for weight matrices
    type int8_gemm_array_t is array (0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1) of signed(7 downto 0);
    signal result_index : integer := 0;

    -- DUT Signals for qkv_proj_calc and gemm_array
    signal clk                : std_logic := '0';
    signal rst                : std_logic := '0';
    signal start              : std_logic := '0';
    signal done               : std_logic;
    signal calc_bram_addr     : integer range 0 to (3 * (HEAD_DIM * MODEL_DIM) + TOKEN_DIM) - 1 := 0;
    signal calc_bram_data_in  : std_logic_vector(7 downto 0)                                    := (others => '0');
    signal calc_bram_data_out : std_logic_vector(7 downto 0);
    signal calc_bram_wr       : std_logic := '0';
    signal input_a            : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal input_b            : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal output_d           : std_logic_vector(32 * TILE_H - 1 downto 0);
    signal start_gemm         : std_logic;
    signal fetch_gemm         : std_logic;
    signal done_gemm          : std_logic := '0';

    -- QKV bram Signals
    signal fetch_qkv          : std_logic                                               := '0';
    signal proj_bram_wr       : std_logic                                               := '0';
    signal proj_bram_rd       : std_logic                                               := '0';
    signal proj_bram_addr     : integer range 0 to QKV_BRAM_ELEM - 1                    := 0;
    signal proj_bram_data_in  : std_logic_vector(TILE_H * QKV_BRAM_DATA_W - 1 downto 0) := (others => '0');
    signal proj_bram_data_out : std_logic_vector(TILE_H * QKV_BRAM_DATA_W - 1 downto 0);

begin
    -- Instantiate Multi-Head Attention Core
    qkv_proj_calc : entity work.qkv_proj_calc
        generic
        map (
        DATA_W => MHA_DATA_W,
        TILE_H => TILE_H,
        TILE_W => TILE_W
        )
        port map
        (
            clk           => clk,
            rst           => rst,
            start         => start,
            done          => done,
            bram_addr     => calc_bram_addr,
            bram_data_in  => calc_bram_data_in,
            bram_data_out => calc_bram_data_out,
            bram_wr       => calc_bram_wr,
            input_a       => input_a,
            input_b       => input_b,
            start_gemm    => start_gemm,
            fetch_gemm    => fetch_gemm,
            done_gemm     => done_gemm
        );

    -- Instantiate Matrix Multiply Core
    gemm_array : entity work.gemm_array
        generic
        map (
        DATA_W => MM_IN_DATA_W,
        TILE_H => TILE_H,
        TILE_W => TILE_W
        )
        port
        map (
        clk      => clk,
        rst      => rst,
        start    => start_gemm,
        clear    => fetch_gemm,
        input_a  => input_a,
        input_b  => input_b,
        output_d => output_d,
        done     => done_gemm
        );

    -- Instantiate QKV bram (DUT)
    qkv_proj_bram_inst : entity work.qkv_proj_bram
        generic
        map (
        DATA_W => QKV_BRAM_DATA_W
        )
        port
        map (
        clk       => clk,
        rst       => rst,
        fetch_qkv => fetch_qkv,
        bram_rd   => proj_bram_rd,
        bram_addr => proj_bram_addr,
        data_in   => proj_bram_data_in,
        data_out  => proj_bram_data_out
        );

    -- Connect matrix multiplication output to bram input
    process (output_d)
    begin
        for i in 0 to TILE_H - 1 loop
            proj_bram_data_in((QKV_BRAM_DATA_W * i + QKV_BRAM_DATA_W) - 1 downto QKV_BRAM_DATA_W * i)
            <= output_d((MM_OUT_DATA_W * i + QKV_BRAM_DATA_W) - 1 downto MM_OUT_DATA_W * i);
        end loop;
    end process;
    -- Synchronize fetch_qkv with fetch_gemm to store QKV data
    fetch_qkv <= fetch_gemm;

    -- Clock generation
    clk_process : process
    begin
        loop
            clk <= '0';
            wait for clk_period / 2;
            clk <= '1';
            wait for clk_period / 2;
        end loop;
    end process;

    rst_process : process
    begin
        -- Initialize reset
        rst <= '1';
        wait for 20 ns;
        rst <= '0';
        wait;
    end process;

    -- Stimulus and verification process
    stim_proc : process
        variable rand_val     : integer;
        variable line_bramfer : line;
        -- Matrices for token embeddings and QKV weights
        variable token_mat : int8_gemm_array(0 to 0, 0 to TOKEN_DIM - 1);
        variable q_mat     : int8_gemm_array(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
        variable k_mat     : int8_gemm_array(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
        variable v_mat     : int8_gemm_array(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
    begin
        -- Process tokens and heads
        for tok_num in 0 to 4 loop -- Test 4 tokens
            read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/token_embeddings_" & integer'image(tok_num) & ".txt",
            1, TOKEN_DIM, token_mat);
            -- Token Embeddings 
            for c in 0 to TOKEN_DIM - 1 loop
                -- Calculate address (last 256 elements)
                calc_bram_addr    <= (3 * HEAD_DIM * MODEL_DIM) + c;
                calc_bram_data_in <= std_logic_vector(token_mat(0, c));
                calc_bram_wr      <= '1';
                wait for clk_period;
            end loop;
            calc_bram_wr <= '0';
            wait for clk_period;
            wait for 2 * clk_period;

            for head_num in 0 to 7 loop -- Test all 8 heads
                -- Load weight matrices for current head
                read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/layer0_query_head" & integer'image(head_num) & ".txt",
                HEAD_DIM, MODEL_DIM, q_mat);
                read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/layer0_key_head" & integer'image(head_num) & ".txt",
                HEAD_DIM, MODEL_DIM, k_mat);
                read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/layer0_value_head" & integer'image(head_num) & ".txt",
                HEAD_DIM, MODEL_DIM, v_mat);

                -- Write Query weights to BRAM
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        calc_bram_addr    <= Q_BASE + r * WEIGHT_SLICE + c;
                        calc_bram_data_in <= std_logic_vector(q_mat(r, c));
                        calc_bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                calc_bram_wr <= '0';
                wait for 2 * clk_period;

                -- Write Key weights to BRAM
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        calc_bram_addr    <= K_BASE + r * WEIGHT_SLICE + c;
                        calc_bram_data_in <= std_logic_vector(k_mat(r, c));
                        calc_bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                calc_bram_wr <= '0';
                wait for 2 * clk_period;

                -- Write Value weights to BRAM
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        calc_bram_addr    <= V_BASE + r * WEIGHT_SLICE + c;
                        calc_bram_data_in <= std_logic_vector(v_mat(r, c));
                        calc_bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                calc_bram_wr <= '0';
                wait for 2 * clk_period;

                -- Start MHA core to compute QKV projections
                start <= '1';
                wait for clk_period;
                start <= '0';

                wait until done = '1';
                wait for 2 * clk_period;
            end loop;
        end loop;
    end process;

end architecture;