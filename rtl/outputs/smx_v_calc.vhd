-- =============================================================================
--  File      : smx_v_calc.vhd  (DRAFT — compile-unverified: no simulator in run)
--  Purpose   : Softmax normalize + attention weighted-sum.
--              Per (head, query row): buffers the row's Q9 exp values, divides
--              each by the row partial sum (from smx_cache) to form Q7 weights
--              w_k = (exp_k << 7) / sum, then accumulates attn[d] = sum_k w_k*V[k][d].
--  Notes     : One shared sequential restoring divider, time-multiplexed across
--              the row's keys (DSP/utilization risk note: 1 divider + 1 mult
--              reused, not duplicated per lane). Compile-unverified draft.
-- =============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

entity smx_v_calc is
    generic
    (
        MAX_TOK : positive := 16;  -- keys per row (context)
        HEAD_DIM: positive := 64;  -- V vector width
        EXP_W   : positive := 32;  -- Q9 exp value width
        SUM_W   : positive := 48;  -- partial-sum width (from smx_cache)
        W_SHIFT : natural  := 7    -- weight Q-format (Q7)
    );
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        -- Exp stream for the current row (same stream smx_cache sees)
        exp_valid : in std_logic;
        exp_last  : in std_logic;
        exp_data  : in signed(EXP_W - 1 downto 0);

        -- Row partial sum, valid when row_sum_valid='1' (from smx_cache rd_sum)
        row_sum       : in signed(SUM_W - 1 downto 0);
        row_sum_valid : in std_logic;

        -- V stream: after weights are ready, one V element per valid cycle,
        -- keyed by (v_key, v_dim)
        v_valid : in std_logic;
        v_key   : in integer range 0 to MAX_TOK - 1;
        v_dim   : in integer range 0 to HEAD_DIM - 1;
        v_data  : in signed(7 downto 0);            -- int8 V element

        -- Attention output stream: one accumulated element per cycle
        attn_valid : out std_logic;
        attn_dim   : out integer range 0 to HEAD_DIM - 1;
        attn_data  : out signed(EXP_W - 1 downto 0);

        done : out std_logic
    );
end entity;

architecture rtl of smx_v_calc is

    type state_type is (collect, normalize, weight_store, mac, drain);
    type exp_buf_type is array (0 to MAX_TOK - 1) of signed(EXP_W - 1 downto 0);
    type w_buf_type is array (0 to MAX_TOK - 1) of signed(EXP_W - 1 downto 0);
    type acc_type is array (0 to HEAD_DIM - 1) of signed(EXP_W + 16 - 1 downto 0);

    signal state    : state_type := collect;
    signal exp_buf  : exp_buf_type := (others => (others => '0'));
    signal w_buf    : w_buf_type := (others => (others => '0'));
    signal acc      : acc_type := (others => (others => '0'));
    signal key_ctr  : integer range 0 to MAX_TOK - 1 := 0;
    signal dim_ctr  : integer range 0 to HEAD_DIM - 1 := 0;
    signal n_keys   : integer range 0 to MAX_TOK := 0;

    -- Restoring divider registers: computes (exp << W_SHIFT) / sum
    signal div_rem  : unsigned(SUM_W - 1 downto 0) := (others => '0');
    signal div_num  : unsigned(SUM_W - 1 downto 0) := (others => '0');
    signal div_den  : unsigned(SUM_W - 1 downto 0) := (others => '0');
    signal div_quot : unsigned(EXP_W + W_SHIFT - 1 downto 0) := (others => '0');
    signal div_ctr  : integer range 0 to EXP_W + W_SHIFT := 0;

begin

    process (clk, rst)
        variable rem_shift : unsigned(SUM_W - 1 downto 0);
    begin
        if rst = '1' then
            state      <= collect;
            key_ctr    <= 0;
            dim_ctr    <= 0;
            n_keys     <= 0;
            attn_valid <= '0';
            done       <= '0';

        elsif rising_edge(clk) then
            attn_valid <= '0';
            done       <= '0';

            case state is

                -- Buffer the row's exp values as they stream past
                when collect =>
                    if exp_valid = '1' then
                        exp_buf(key_ctr) <= exp_data;
                        if exp_last = '1' then
                            n_keys  <= key_ctr + 1;
                            key_ctr <= 0;
                            if row_sum_valid = '1' and row_sum > 0 then
                                -- Start divider on key 0
                                div_num  <= unsigned(resize(shift_left(resize(exp_buf(0), SUM_W), W_SHIFT), SUM_W));
                                div_den  <= unsigned(row_sum);
                                div_rem  <= (others => '0');
                                div_quot <= (others => '0');
                                div_ctr  <= 0;
                                state    <= normalize;
                            end if;
                        else
                            key_ctr <= key_ctr + 1;
                        end if;
                    end if;

                -- Sequential restoring division, 1 bit per cycle, per key
                when normalize =>
                    rem_shift := div_rem(SUM_W - 2 downto 0) & div_num(SUM_W - 1);
                    div_num   <= div_num(SUM_W - 2 downto 0) & '0';
                    if rem_shift >= div_den then
                        div_rem  <= rem_shift - div_den;
                        div_quot <= div_quot(div_quot'left - 1 downto 0) & '1';
                    else
                        div_rem  <= rem_shift;
                        div_quot <= div_quot(div_quot'left - 1 downto 0) & '0';
                    end if;

                    if div_ctr = EXP_W + W_SHIFT - 1 then
                        w_buf(key_ctr) <= signed(resize(div_quot, EXP_W));
                        div_ctr        <= 0;
                        if key_ctr = n_keys - 1 then
                            key_ctr <= 0;
                            state   <= mac;
                        else
                            key_ctr <= key_ctr + 1;
                            -- Load next key's numerator
                            div_num <= unsigned(resize(shift_left(resize(exp_buf(key_ctr + 1), SUM_W), W_SHIFT), SUM_W));
                            div_rem <= (others => '0');
                        end if;
                    else
                        div_ctr <= div_ctr + 1;
                    end if;

                -- Weighted V accumulation: attn[d] += w[key] * V[key][d]
                when mac =>
                    if v_valid = '1' then
                        acc(v_dim) <= acc(v_dim) + resize(w_buf(v_key) * v_data, acc(v_dim)'length);
                        if v_key = n_keys - 1 and v_dim = HEAD_DIM - 1 then
                            state <= drain;
                        end if;
                    end if;

                -- Stream results out, >> W_SHIFT back to element scale
                when drain =>
                    attn_valid <= '1';
                    attn_dim   <= dim_ctr;
                    attn_data  <= resize(shift_right(acc(dim_ctr), W_SHIFT), EXP_W);
                    if dim_ctr = HEAD_DIM - 1 then
                        dim_ctr <= 0;
                        done    <= '1';
                        state   <= collect;
                        acc     <= (others => (others => '0'));
                    else
                        dim_ctr <= dim_ctr + 1;
                    end if;

                when others =>
                    state <= collect;
            end case;
        end if;
    end process;

end architecture;
