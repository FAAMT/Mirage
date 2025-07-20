-- ============================================================================
--  File      : qkv_score_calc_tb.vhd
--  Author    : Fahad Tajiki
--  Date      : 10/07/2025
--  Purpose   : Testbench for QKV Score Calculation module, verifying correct
--              computation of attention scores from Query and Key vectors.
--              Integrates with QKV BRAM to fetch vectors and validate scores.
-- ============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use STD.TEXTIO.all;
use IEEE.STD_LOGIC_TEXTIO.all;
use work.types_pkg.all;
-- use work.tb_io_pkg.all;

entity qkv_score_calc_tb is
end entity;

architecture behavior of qkv_score_calc_tb is
    -- Array to store expected attention scores
    type score_array is array (0 to 15, 0 to 15) of std_logic_vector(31 downto 0);
    signal expected_scores : score_array := (others => (others => (others => '0')));

    -- CLK & RST Signals
    signal clk : std_logic := '0';
    signal rst : std_logic := '0';

    -- UUT Signals for qkv_score_calc
    signal score_start         : std_logic := '0';
    signal score_done          : std_logic;
    signal score_bram_addr     : integer range 0 to QKV_BRAM_ELEM - 1                    := 0;
    signal score_bram_data_in  : std_logic_vector(TILE_H * QKV_BRAM_DATA_W - 1 downto 0) := (others => '0');
    signal score_bram_data_out : std_logic_vector(TILE_H * QKV_BRAM_DATA_W - 1 downto 0);
    signal score_bram_rd       : std_logic := '0';
    signal score_start_gemm    : std_logic;
    signal score_clear_gemm    : std_logic;
    signal score_done_gemm     : std_logic := '0';
    signal fetch_qkv_i         : std_logic := '0';

    -- Test Module Signals for qkv_proj_calc
    signal proj_start         : std_logic := '0';
    signal proj_done          : std_logic;
    signal proj_bram_addr     : integer range 0 to BASE_OFFSET + TOKEN_DIM - 1 := 0;
    signal proj_bram_data_in  : std_logic_vector(MHA_DATA_W - 1 downto 0)      := (others => '0');
    signal proj_bram_data_out : std_logic_vector(MHA_DATA_W - 1 downto 0);
    signal proj_bram_wr       : std_logic := '0';
    signal proj_bram_rd       : std_logic := '0';
    signal proj_start_gemm    : std_logic;
    signal proj_fetch_gemm    : std_logic;
    signal proj_done_gemm     : std_logic := '0';

    -- Test Module SIgnals for gemm_array
    signal start_gemm_i : std_logic := '0';
    signal clear_gemm_i : std_logic := '0';
    signal fetch_gemm_i : std_logic := '0';
    signal done_gemm_i  : std_logic := '0';

    -- Arbiter Signals
    signal proj_input_a  : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal proj_input_b  : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal score_input_a : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal score_input_b : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);

    -- GEMM Array Internal Signals
    signal input_a_i  : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal input_b_i  : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal output_d_i : std_logic_vector(32 * TILE_H - 1 downto 0);

begin
    -- Instantiate QKV Score Calculation Core (DUT)
    qkv_score_calc : entity work.qkv_score_calc
        generic
        map (
        QKV_DATA_W   => QKV_BRAM_DATA_W,
        SCORE_DATA_W => SCORE_BRAM_DATA_W
        )
        port map
        (
            clk   => clk,
            rst   => rst,
            start => score_start,
            done  => score_done,

            -- Proj. BRAM Intrfc
            bram_rd      => score_bram_rd,
            bram_addr    => score_bram_addr,
            bram_data_in => score_bram_data_in,

            -- GEMM Array Intrfc
            start_gemm => score_start_gemm,
            clear_gemm => score_clear_gemm,
            done_gemm  => score_done_gemm,

            input_a  => score_input_a,
            input_b  => score_input_b,
            output_d => output_d_i,

            fetch_qkv => proj_fetch_gemm
        );

    -- Instantiate Multi-Head Attention Core
    qkv_proj_calc : entity work.qkv_proj_calc
        generic
        map (
        DATA_W => MHA_DATA_W,
        TILE_H => TILE_H,
        TILE_W => TILE_W
        )
        port
        map
        (
        clk   => clk,
        rst   => rst,
        start => proj_start,
        done  => proj_done,

        -- Token, Bias, & Weights BRAM Intrfc
        bram_addr     => proj_bram_addr,
        bram_data_in  => proj_bram_data_in,
        bram_data_out => proj_bram_data_out,
        bram_wr       => proj_bram_wr,

        -- GEMM Array Intrfc
        start_gemm => proj_start_gemm,
        done_gemm  => proj_done_gemm,

        input_a => proj_input_a,
        input_b => proj_input_b,

        -- Additional Output
        fetch_gemm => proj_fetch_gemm
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
        start    => start_gemm_i,
        clear    => proj_fetch_gemm or score_clear_gemm,
        input_a  => input_a_i,
        input_b  => input_b_i,
        output_d => output_d_i,
        done     => done_gemm_i
        );

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

    -- -- Arbiter process to manage shared resources
    -- arbiter_proc : process (clk, rst)
    -- begin
    --     if rst = '1' then
    --         -- Default assignments
    --         start_gemm_i <= '0';

    --         input_a_i <= (others => (others => (others => '0')));
    --         input_b_i <= (others => (others => (others => '0')));

    --     elsif rising_edge(clk) then
    --         -- Priority goes to projection calculation when active
    --         if proj_start = '1' then
    --             -- Connect projection calc signals
    --             start_gemm_i <= proj_start_gemm;
    --             fetch_gemm_i <= proj_fetch_gemm;
    --             input_a_i    <= proj_input_a;
    --             input_b_i    <= proj_input_b;

    --             -- Feedback to projection calc
    --             proj_done_gemm <= done_gemm_i;

    --             -- Populate Proj. BRAM inside of QKV Score Calc module => .
    --             for i in 0 to TILE_H - 1 loop
    --                 score_bram_data_in((QKV_BRAM_DATA_W * i + QKV_BRAM_DATA_W) - 1 downto QKV_BRAM_DATA_W * i)
    --                 <= output_d_i((MM_OUT_DATA_W * i + QKV_BRAM_DATA_W) - 1 downto MM_OUT_DATA_W * i);
    --             end loop;
    --             proj_bram_data_out <= proj_bram_data_out;

    --             -- When projection calc is idle, score calc can use resources
    --         elsif score_start = '1' then
    --             -- Connect score calc signals
    --             start_gemm_i <= score_start_gemm;
    --             fetch_gemm_i <= score_clear_gemm;
    --             input_a_i    <= score_input_a;
    --             input_b_i    <= score_input_b;

    --             -- Feedback to score calc
    --             score_done_gemm <= done_gemm_i;
    --             -- score_bram_data_out <= proj_bram_data_out;
    --         end if;
    --     end if;
    -- end process;

    -- Combinational arbiter process to manage shared resources
    arbiter_proc : process (all)
    begin
        -- Default assignments
        start_gemm_i    <= '0';
        fetch_gemm_i    <= '0';
        input_a_i       <= (others => (others => (others => '0')));
        input_b_i       <= (others => (others => (others => '0')));
        proj_done_gemm  <= '0';
        score_done_gemm <= '0';

        -- Connect output data to score BRAM input
        for i in 0 to TILE_H - 1 loop
            score_bram_data_in((QKV_BRAM_DATA_W * i + QKV_BRAM_DATA_W) - 1 downto QKV_BRAM_DATA_W * i)
            <= output_d_i((MM_OUT_DATA_W * i + QKV_BRAM_DATA_W) - 1 downto MM_OUT_DATA_W * i);
        end loop;

        -- Priority goes to projection calculation when active
        if proj_start = '1' then
            -- Connect projection calc signals
            start_gemm_i   <= proj_start_gemm;
            fetch_gemm_i   <= proj_fetch_gemm;
            input_a_i      <= proj_input_a;
            input_b_i      <= proj_input_b;
            proj_done_gemm <= done_gemm_i;

            -- When projection calc is idle, score calc can use resources
        elsif score_start = '1' then
            -- Connect score calc signals
            start_gemm_i    <= score_start_gemm;
            fetch_gemm_i    <= score_clear_gemm;
            input_a_i       <= score_input_a;
            input_b_i       <= score_input_b;
            score_done_gemm <= done_gemm_i;
        end if;
    end process;

    -- Connect matrix multiplication output to bram input
    -- process (output_d_i)
    -- begin
    --     for i in 0 to TILE_H - 1 loop
    --         proj_bram_data_in((QKV_BRAM_DATA_W * i + QKV_BRAM_DATA_W) - 1 downto QKV_BRAM_DATA_W * i)
    --         <= output_d_i((MM_OUT_DATA_W * i + QKV_BRAM_DATA_W) - 1 downto MM_OUT_DATA_W * i);
    --     end loop;
    -- end process;

    -- Stimulus process
    stim_proc : process
        ---------------------------------------------------------
        -- ▶ Load token embeddings & QKV weights from text files --
        ---------------------------------------------------------
        variable token_mat : int8_gemm_array(0 to 0, 0 to TOKEN_DIM - 1);
        variable q_mat     : int8_gemm_array(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
        variable k_mat     : int8_gemm_array(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
        variable v_mat     : int8_gemm_array(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
    begin
        for tok_num in 0 to 0 loop
            -- Process all 8 heads in a loop
            read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/token_embeddings_" & integer'image(tok_num) & ".txt",
            1, TOKEN_DIM, token_mat);

            -------------------------------------------------------------------
            -- ▶ 1. Token Embeddings 
            -------------------------------------------------------------------
            for c in 0 to TOKEN_DIM - 1 loop
                -- Calculate address (last 256 elements)
                proj_bram_addr    <= BASE_OFFSET + c;
                proj_bram_data_in <= std_logic_vector(token_mat(0, c));
                proj_bram_wr      <= '1';
                wait for clk_period;
            end loop;
            proj_bram_wr <= '0';
            wait for clk_period;
            wait for 2 * clk_period;

            for head_num in 0 to 7 loop
                -- Read weight matrices for current head
                read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/layer0_query_head" & integer'image(head_num) & ".txt",
                HEAD_DIM, MODEL_DIM, q_mat);
                read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/layer0_key_head" & integer'image(head_num) & ".txt",
                HEAD_DIM, MODEL_DIM, k_mat);
                read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/layer0_value_head" & integer'image(head_num) & ".txt",
                HEAD_DIM, MODEL_DIM, v_mat);

                -------------------------------------------------------------------
                -- ▶ 2. Q weights (layer-0, head-N) 0 … 16 383
                -------------------------------------------------------------------
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        proj_bram_addr    <= Q_BASE + r * WEIGHT_SLICE + c;
                        proj_bram_data_in <= std_logic_vector(q_mat(r, c));
                        proj_bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                proj_bram_wr <= '0';
                wait for 2 * clk_period;

                -------------------------------------------------------------------
                -- ▶ 3. K weights 16 384 … 32 767
                -------------------------------------------------------------------
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        proj_bram_addr    <= K_BASE + r * WEIGHT_SLICE + c;
                        proj_bram_data_in <= std_logic_vector(k_mat(r, c));
                        proj_bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                proj_bram_wr <= '0';
                wait for 2 * clk_period;

                -------------------------------------------------------------------
                -- ▶ 4. V weights 32 768 … 49 151
                -------------------------------------------------------------------
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        proj_bram_addr    <= V_BASE + r * WEIGHT_SLICE + c;
                        proj_bram_data_in <= std_logic_vector(v_mat(r, c));
                        proj_bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                proj_bram_wr <= '0';
                wait for 2 * clk_period;

                -- Start the core
                proj_start <= '1';
                wait until proj_done = '1';
                proj_start <= '0';
                wait for 2 * clk_period;
            end loop;

            -- Start the QKV Score Calc core
            score_start            <= '1';
            wait until score_start <= '0';
            wait for 2 * clk_period;

        end loop;

        wait for 10 * clk_period; -- small stabilization
        wait;
    end process;
end architecture;