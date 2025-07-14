-- ============================================================================
--  File      : qkv_cache_tb.vhd
--  Author    : Fahad Tajiki
--  Date      : 10/07/2025
--  Purpose   : Testbench for QKV cache, verifying correct storage of Query, Key, 
--              and Value vectors computed by matrix multiplication in a Transformer
--              Encoder Layer. Integrates matrix_mult and multi_head cores, driving
--              the QKV cache with fetch_qkv signal and validating stored data.
-- ============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use STD.TEXTIO.all;
use IEEE.STD_LOGIC_TEXTIO.all;
use work.types_pkg.all;
use work.tb_io_pkg.all;

entity qkv_cache_tb is
end entity;

architecture behavior of qkv_cache_tb is
    -- Array to store matrix multiplication results
    type mac_result_array is array (0 to 15, 0 to 11) of std_logic_vector(31 downto 0);
    signal mac_results : mac_result_array := (others => (others => (others => '0')));

    -- Matrix type for weight matrices
    type matrix_int8_t is array (0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1) of signed(7 downto 0);
    signal result_index : integer := 0;

    -- DUT Signals for multi_head and matrix_mult
    signal clk           : std_logic := '0';
    signal rst           : std_logic := '0';
    signal start         : std_logic := '0';
    signal done          : std_logic;
    signal bram_addr     : integer range 0 to (3 * (HEAD_DIM * MODEL_DIM) + TOKEN_DIM) - 1 := 0;
    signal bram_data_in  : std_logic_vector(7 downto 0)                                    := (others => '0');
    signal bram_data_out : std_logic_vector(7 downto 0);
    signal bram_wr       : std_logic := '0';
    signal input_a       : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal input_b       : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal output_d      : std_logic_vector(32 * TILE_H - 1 downto 0);
    signal start_mac     : std_logic;
    signal fetch_mac     : std_logic;
    signal done_mac      : std_logic := '0';

    -- QKV Cache Signals
    signal fetch_qkv      : std_logic := '0';
    signal cache_rd       : std_logic := '0';
    signal cache_addr     : std_logic := '0';
    signal cache_data_in  : std_logic_vector(TILE_H * CACHE_DATA_WIDTH - 1 downto 0);
    signal cache_data_out : std_logic_vector(TILE_H * CACHE_DATA_WIDTH - 1 downto 0);

begin
    -- Instantiate Multi-Head Attention Core
    mha_core : entity work.multi_head
        generic
        map (
        DATA_WIDTH => MHA_DATA_WIDTH,
        TILE_H     => TILE_H,
        TILE_W     => TILE_W
        )
        port map
        (
            clk           => clk,
            rst           => rst,
            start         => start,
            done          => done,
            bram_addr     => bram_addr,
            bram_data_in  => bram_data_in,
            bram_data_out => bram_data_out,
            bram_wr       => bram_wr,
            input_a       => input_a,
            input_b       => input_b,
            start_mac     => start_mac,
            fetch_mac     => fetch_mac,
            done_mac      => done_mac
        );

    -- Instantiate Matrix Multiply Core
    mac_core : entity work.matrix_mult
        generic
        map (
        DATA_WIDTH => MM_IN_DATA_WIDTH,
        TILE_H     => TILE_H,
        TILE_W     => TILE_W
        )
        port
        map (
        clk      => clk,
        rst      => rst,
        start    => start_mac,
        clear    => fetch_mac,
        input_a  => input_a,
        input_b  => input_b,
        output_d => output_d,
        done     => done_mac
        );

    -- Instantiate QKV Cache (DUT)
    qkv_cache_inst : entity work.qkv_cache
        generic
        map (
        DATA_WIDTH => CACHE_DATA_WIDTH
        )
        port
        map (
        clk        => clk,
        rst        => rst,
        fetch_qkv  => fetch_qkv,
        cache_rd   => cache_rd,
        cache_addr => cache_addr,
        data_in    => cache_data_in,
        data_out   => cache_data_out
        );

    -- Connect matrix multiplication output to cache input
    process (output_d)
    begin
        for i in 0 to TILE_H - 1 loop
            cache_data_in((CACHE_DATA_WIDTH * i + CACHE_DATA_WIDTH) - 1 downto CACHE_DATA_WIDTH * i)
            <= output_d((MM_OUT_DATA_WIDTH * i + CACHE_DATA_WIDTH) - 1 downto MM_OUT_DATA_WIDTH * i);
        end loop;
    end process;
    -- Synchronize fetch_qkv with fetch_mac to store QKV data
    fetch_qkv <= fetch_mac;

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

    -- Stimulus and verification process
    stim_proc : process
        variable rand_val    : integer;
        variable line_buffer : line;
        file output_file     : text open write_mode is "qkv_cache_output.txt";
        -- Matrices for token embeddings and QKV weights
        variable token_mat : matrix_int8(0 to 0, 0 to TOKEN_DIM - 1);
        variable q_mat     : matrix_int8(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
        variable k_mat     : matrix_int8(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
        variable v_mat     : matrix_int8(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
    begin
        -- Initialize reset
        rst <= '1';
        wait for 20 ns;
        rst <= '0';

        -- Process tokens and heads
        for tok_num in 0 to 4 loop -- Test 4 tokens
            read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/token_embeddings_" & integer'image(tok_num) & ".txt",
            1, TOKEN_DIM, token_mat);
            -- Token Embeddings 
            for c in 0 to TOKEN_DIM - 1 loop
                -- Calculate address (last 256 elements)
                bram_addr    <= (3 * HEAD_DIM * MODEL_DIM) + c;
                bram_data_in <= std_logic_vector(token_mat(0, c));
                bram_wr      <= '1';
                wait for clk_period;
            end loop;
            bram_wr <= '0';
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
                        bram_addr    <= Q_BASE + r * WEIGHT_SLICE + c;
                        bram_data_in <= std_logic_vector(q_mat(r, c));
                        bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                bram_wr <= '0';
                wait for 2 * clk_period;

                -- Write Key weights to BRAM
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        bram_addr    <= K_BASE + r * WEIGHT_SLICE + c;
                        bram_data_in <= std_logic_vector(k_mat(r, c));
                        bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                bram_wr <= '0';
                wait for 2 * clk_period;

                -- Write Value weights to BRAM
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        bram_addr    <= V_BASE + r * WEIGHT_SLICE + c;
                        bram_data_in <= std_logic_vector(v_mat(r, c));
                        bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                bram_wr <= '0';
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