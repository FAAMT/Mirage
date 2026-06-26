-- =============================================================================
--  File      : qk_score_bram.vhd
--  Author    : Fahad Tajiki
--  Date      : 31/07/2025
--  Purpose   : Implements an on-chip bram for Query, Key, and Value (QKV) vectors
--              in a Transformer Encoder Layer on FPGA. Stores 128 elements representing 
--              4 tokens per head per Q & K pair asa score. This represents <4x<4x8>>
--              output stream per vatch (size=4) of tokens.
-- ============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity qk_score_bram is
    generic
    (
        -- Width of data elements in bram
        DATA_W : integer := SCORE_BRAM_DATA_W
    );
    port
    (
        -- Clock and reset signals
        clk : in std_logic;
        rst : in std_logic;

        -- Control signal to initiate QK^T score fetch
        fetch_score : in std_logic;
        qk_iter     : in integer;
        tok_q_iter  : in integer;
        tok_k_iter  : in integer;
        -- BRAM Interface
        bram_rd    : in std_logic;
        addr_off_0 : in integer;
        addr_off_1 : in integer;
        -- IO Data Stream 
        bram_data_in  : in std_logic_vector(32 * TILE_H - 1 downto 0);
        bram_data_out : out score_io_type
    );
end entity;

architecture rtl of qk_score_bram is
    type state_type is (idle, fetch_scores);
    -- Current and previous state registers
    signal state      : state_type := idle;
    signal fetch_ctr  : integer    := 0;
    signal batch_iter : integer    := 0;
    -- Block RAM array for QKV data
    signal bram_score : bram_scores_type;
    -- Address for accessing QKV bram
    signal bram_score_addr : integer range 0 to SCORE_BRAM_ELEM - 1 := 0;

    signal qk_iter_reg    : integer;
    signal tok_q_iter_reg : integer;
    signal tok_k_iter_reg : integer;

begin
    -- BRAM read process
    process (addr_off_0, bram_score)
    begin
        for i in 0 to TOK_CONTEXT * NUM_HEADS - 1 loop
            bram_data_out(i) <= std_logic_vector(bram_score(addr_off_0 + i));
        end loop;
    end process;

    process (clk, rst)
    begin
        if rst = '1' then
            qk_iter_reg    <= 0;
            tok_q_iter_reg <= 0;
            tok_k_iter_reg <= 0;

        elsif rising_edge(clk) then
            qk_iter_reg    <= qk_iter;
            tok_q_iter_reg <= tok_q_iter;
            tok_k_iter_reg <= tok_k_iter;
        end if;
    end process;

    -- Address calculation for QKV bram access
    process (state, qk_iter_reg, tok_q_iter_reg, tok_k_iter_reg)
        variable addr_assign : integer := 0;
    begin
        case state is
            when fetch_scores =>
                -- Calculate address for Query vector
                if tok_q_iter_reg > 0 then
                    if qk_iter_reg > 0 then
                        if tok_k_iter_reg > 0 then
                            addr_assign :=
                                (tok_k_iter_reg) +
                                TOK_CONTEXT * (qk_iter_reg) +
                                TOK_CONTEXT * NUM_HEADS * (tok_q_iter_reg);
                        else
                            addr_assign :=
                                TOK_CONTEXT * (qk_iter_reg) +
                                TOK_CONTEXT * NUM_HEADS * (tok_q_iter_reg);
                        end if;
                    else
                        if tok_k_iter_reg > 0 then
                            addr_assign :=
                                (tok_k_iter_reg) +
                                TOK_CONTEXT * NUM_HEADS * (tok_q_iter_reg);
                        else
                            addr_assign :=
                                TOK_CONTEXT * NUM_HEADS * (tok_q_iter_reg);
                        end if;
                    end if;
                else
                    if qk_iter_reg > 0 then
                        if tok_k_iter_reg > 0 then
                            addr_assign :=
                                (tok_k_iter_reg) +
                                TOK_CONTEXT * (qk_iter_reg);
                        else
                            addr_assign :=
                                TOK_CONTEXT * (qk_iter_reg);
                        end if;
                    else
                        if tok_k_iter_reg > 0 then
                            addr_assign :=
                                (tok_k_iter_reg);
                        else
                            addr_assign := 0;
                        end if;
                    end if;
                end if;

                if addr_assign < SCORE_BRAM_ELEM then
                    bram_score_addr <= addr_assign;
                end if;

            when others =>
                -- Default to zero address in idle state
                bram_score_addr <= 0;
        end case;
    end process;

    -- Main control and data storage process
    process (clk, rst)
        variable score_sum : signed(DATA_W - 1 downto 0) := (others => '0');
    begin
        if rst = '1' then
            -- Reset all control and data signals
            state      <= idle;
            batch_iter <= 0;
            fetch_ctr  <= 0;
            bram_score <= (others => (others => '0'));
        elsif rising_edge(clk) then
            case state is
                when idle =>
                    state <= fetch_scores;

                when fetch_scores =>
                    if fetch_score = '1' then
                        -- Store 8 qk pairs per head then increment to next token [0-3] in batch
                        if fetch_ctr = NUM_HEADS - 1 then
                            batch_iter <= batch_iter + 1;
                            fetch_ctr  <= 0;
                        elsif fetch_ctr < NUM_HEADS - 1 then
                            fetch_ctr <= fetch_ctr + 1;
                        end if;

                        -- Calculate the sum of all 32-bit segments in output_d
                        for i in 0 to 11 loop
                            score_sum := score_sum + resize(signed(bram_data_in(32 * i + 31 downto 32 * i)), DATA_W);
                        end loop;

                        bram_score(bram_score_addr) <= score_sum;
                    else
                        score_sum := to_signed(0, DATA_W); -- Reset Score Sum
                    end if;
            end case;
        end if;
    end process;

end architecture;