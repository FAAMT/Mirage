library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity softmax is
    generic
    (
        DATA_WIDTH : integer := 16
    );
    port
    (
        clk        : in std_logic;
        rst        : in std_logic;
        start      : in std_logic;
        done       : out std_logic;
        fetch_qkv  : in std_logic;
        fetch_attn : out std_logic;
        data_in    : in std_logic_vector(12 * 16 - 1 downto 0);
        data_out   : out std_logic_vector(12 * DATA_WIDTH - 1 downto 0);

        -- Matrix Mult Control Interface
        start_mac : out std_logic;
        clear_mac : out std_logic;
        done_mac  : in std_logic;

        -- Matrix Mult Data Interface
        input_a  : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        input_b  : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        output_d : in std_logic_vector(32 * TILE_H - 1 downto 0)
    );
end entity;

architecture rtl of softmax is

    -------------------------------------------------------------------
    -- 1) TYPE DECLARATIONS
    -------------------------------------------------------------------
    type state_type is (
        idle, fetch, load,
        compute_scores, compute_softmax,
        compute_attention, result
    );
    type bram_qkv_type is array(0 to 767) of std_logic_vector(15 downto 0);
    type bram_scores_type is array(0 to 15) of signed(17 downto 0);
    type bram_softmax_type is array(0 to 15) of signed(10 downto 0);
    -- type bram_attention_type is array(0 to 15) of signed(17 downto 0);
    type lut_max_type is array(0 to 3) of signed(17 downto 0);

    type taylor_array_18 is array(0 to 3) of signed(17 downto 0);
    type taylor_array_q15 is array(0 to 3) of signed(15 downto 0);
    type taylor_array_q30 is array(0 to 3) of signed(31 downto 0);
    type taylor_array_q45 is array(0 to 3) of signed(47 downto 0);
    type taylor_array_approx is array(0 to 3) of signed(63 downto 0);
    type softmax_array_out is array(0 to 3) of signed(63 downto 0);

    -------------------------------------------------------------------
    -- 2) SIGNAL DECLARATIONS (exactly as you provided)
    -------------------------------------------------------------------
    signal state      : state_type := idle;
    signal prev_state : state_type := idle;

    signal fetch_ctr  : integer range 0 to 5 := 0;
    signal fetch_reg  : std_logic;
    signal start_max  : std_logic;
    signal pause_mac  : std_logic;
    signal q_ext_iter : integer;
    signal q_int_iter : integer;
    signal v_ext_iter : integer;
    signal v_int_iter : integer;

    signal bram_qkv      : bram_qkv_type;
    signal bram_qkv_addr : integer range 0 to 767 := 0;
    signal bram_qkv_off  : integer                := 16;
    signal offset        : integer                := 0;
    signal mac_out_reg   : std_logic_vector(12 * 16 - 1 downto 0);

    signal bram_scores  : bram_scores_type;
    signal bram_softmax : bram_softmax_type;

    signal max_lut     : lut_max_type;
    signal max_val     : signed(17 downto 0);
    signal max_index   : integer range 0 to 3 := 0;
    signal max_ctr     : integer              := 0;
    signal delay_ctr   : integer              := 0;
    signal current_max : signed(17 downto 0);
    signal index       : integer range 0 to 63;
    signal processing  : std_logic;
    signal find_max    : std_logic;

    signal taylor_calc   : std_logic;
    signal taylor_done   : std_logic;
    signal z_diff        : taylor_array_18     := (others => (others => '0'));
    signal z_diff_q15    : taylor_array_q15    := (others => (others => '0'));
    signal z_diff_sq     : taylor_array_q30    := (others => (others => '0'));
    signal z_diff_cu     : taylor_array_q45    := (others => (others => '0'));
    signal term1         : taylor_array_approx := (others => (others => '0'));
    signal term2         : taylor_array_approx := (others => (others => '0'));
    signal term3         : taylor_array_approx := (others => (others => '0'));
    signal taylor_approx : taylor_array_approx := (others => (others => '0'));
    signal taylor_sum    : signed(63 downto 0);

    signal softmax_ready : std_logic;
    signal softmax_ctr   : integer;
    signal softmax_out   : softmax_array_out;

    signal store_attn : std_logic;

begin

    -------------------------------------------------------------------
    -- 3) MAIN FSM
    -------------------------------------------------------------------
    process (clk, rst)
        variable k_idx, r_mod_4, r_mod_3 : integer := 0;
    begin
        if rst = '1' then
            -- Control Signals
            state       <= idle;
            done        <= '0';
            start_max   <= '0';
            delay_ctr   <= 0;
            max_ctr     <= 0;
            taylor_calc <= '0';
            taylor_done <= '0';
            start_mac   <= '0';
            clear_mac   <= '0';
            pause_mac   <= '0';
            store_attn  <= '0';

            -- Loop Signals
            q_ext_iter  <= 0;
            q_int_iter  <= 0;
            v_ext_iter  <= 0;
            v_int_iter  <= 0;
            softmax_ctr <= 0;

            -- BRAM Signals
            bram_qkv      <= (others => (others => '0'));
            bram_scores   <= (others => (others => '0'));
            bram_softmax  <= (others => (others => '0'));
            bram_qkv_addr <= 0;
            offset        <= 0;

            -- Data In & Out Signal
            mac_out_reg <= (others => '0');
            data_out    <= (others => '0');

        elsif rising_edge(clk) then
            case state is

                    -----------------------------------------------------------------
                    -- IDLE: wait for start
                    -----------------------------------------------------------------
                when idle =>
                    -- Reset Loop Iters
                    q_ext_iter <= 0;
                    q_int_iter <= 0;
                    v_ext_iter <= 0;
                    v_int_iter <= 0;

                    -- Reset MAC Controls
                    start_mac <= '0';
                    clear_mac <= '0';
                    pause_mac <= '0';

                    -- Reset MAX & Taylor Controls
                    fetch_reg   <= '0';
                    start_max   <= '0';
                    taylor_calc <= '0';
                    taylor_done <= '0';
                    store_attn  <= '0';

                    -- Reset Counters
                    delay_ctr   <= 0;
                    max_ctr     <= 0;
                    softmax_ctr <= 0;

                    -- Reset Done Signals
                    done <= '0';

                    if start = '1' then
                        state  <= fetch;
                        offset <= 0;
                    end if;

                    -----------------------------------------------------------------
                    -- FETCH: arm for data_in
                    -----------------------------------------------------------------
                when fetch =>
                    -- fetch_reg <= fetch_qkv;
                    start_max   <= '0';
                    mac_out_reg <= data_in;

                    if fetch_qkv = '1' then
                        -- state      <= load;
                        -- prev_state <= fetch;

                        if bram_qkv_addr = 71 then
                            state         <= compute_scores;
                            bram_qkv_addr <= 0;
                        else
                            -- state <= prev_state;
                        end if;

                        bram_qkv_addr <= bram_qkv_addr + 1;

                        for i in 0 to 11 loop
                            if i < 4 then
                                case bram_qkv_addr is
                                    when 5      => offset  <= - 8;
                                    when 11     => offset <= - 16;
                                    when 17     => offset <= - 24;
                                    when 23     => offset <= - 32;
                                    when 29     => offset <= - 40;
                                    when 35     => offset <= - 48;
                                    when 41     => offset <= - 56;
                                    when 47     => offset <= - 64;
                                    when 53     => offset <= - 72;
                                    when 59     => offset <= - 80;
                                    when 65     => offset <= - 88;
                                    when others => null;
                                end case;
                            end if;

                            -- Assign data if within valid range
                            if (bram_qkv_addr = 5 or bram_qkv_addr = 11 or bram_qkv_addr = 17 or bram_qkv_addr = 23 or bram_qkv_addr = 29 or bram_qkv_addr = 35 or bram_qkv_addr = 41 or bram_qkv_addr = 47 or bram_qkv_addr = 53 or bram_qkv_addr = 59 or bram_qkv_addr = 65 or bram_qkv_addr = 71) and i < 4 then
                                bram_qkv(12 * bram_qkv_addr + i + offset) <= data_in((bram_qkv_off * (i + 1) - 1) downto (bram_qkv_off * i));
                            elsif (bram_qkv_addr /= 5 and bram_qkv_addr /= 11 and bram_qkv_addr /= 17 and bram_qkv_addr /= 23 and bram_qkv_addr /= 29 and bram_qkv_addr /= 35 and bram_qkv_addr /= 41 and bram_qkv_addr /= 47 and bram_qkv_addr /= 53 and bram_qkv_addr /= 59 and bram_qkv_addr /= 65 and bram_qkv_addr /= 71) then
                                bram_qkv(12 * bram_qkv_addr + i + offset) <= data_in((bram_qkv_off * (i + 1) - 1) downto (bram_qkv_off * i));
                            end if;
                        end loop;
                    end if;

                    -----------------------------------------------------------------
                    -- COMPUTE_SCORES: issue MAC passes and collect 4×4 results
                    -----------------------------------------------------------------
                when compute_scores =>
                    if q_ext_iter < 4 then
                        if q_int_iter < 5 then
                            if start_mac = '0' and done_mac = '1' then
                                q_int_iter <= q_int_iter + 1;
                            elsif start_mac = '0' and done_mac = '0' then
                                start_mac <= '1';
                            else
                                start_mac <= '0';
                            end if;
                            clear_mac <= '0';
                        else
                            q_ext_iter <= q_ext_iter + 1;
                            q_int_iter <= 0;
                            clear_mac  <= '1';
                            for idx in 0 to 3 loop
                                bram_scores(idx + 4 * q_ext_iter) <=
                                resize(signed(output_d((95 + 96 * idx) downto (64 + 96 * idx))), 18) +
                                resize(signed(output_d((63 + 96 * idx) downto (32 + 96 * idx))), 18) +
                                resize(signed(output_d((31 + 96 * idx) downto (0 + 96 * idx))), 18);
                            end loop;
                        end if;
                    else
                        state      <= compute_softmax;
                        start_mac  <= '0';
                        clear_mac  <= '0';
                        q_ext_iter <= 0;
                        q_int_iter <= 0;
                    end if;

                    for row in 0 to 11 loop
                        for col in 0 to 4 loop
                            k_idx   := row / 3;
                            r_mod_4 := row mod 3;

                            if q_ext_iter < 4 then
                                if q_int_iter < 4 then
                                    -- Q vector input: assume each Q_n is 12×5, stored row-major
                                    input_a(row, col) <= resize(shift_right(signed(bram_qkv(192 * q_ext_iter + 5 * r_mod_4 + col + 15 * q_int_iter)), 8), 8); -- 192 spaces within BRAM

                                    -- K vector input: break rows into 3-row blocks for K0..K3
                                    input_b(row, col) <= resize(shift_right(signed(bram_qkv(64 + 5 * r_mod_4 + col + 192 * k_idx + 15 * q_int_iter)), 8), 8);
                                elsif q_int_iter = 4 then
                                    -- Final pass: map last 4 Q·Kᵀ elements to row 0, cols 0–3
                                    if (row = 0 or row = 3 or row = 6 or row = 9) and col < 4 then
                                        input_a(row, col) <= resize(shift_right(
                                        signed(bram_qkv(192 * q_ext_iter + 15 * q_int_iter + col)), 8), 8); -- Q[15 * q_int_iter + col]

                                        input_b(row, col) <= resize(shift_right(
                                        signed(bram_qkv(64 + 192 * k_idx + 15 * q_int_iter + col)), 8), 8); -- K[15 * q_int_iter + col]
                                    else
                                        input_a(row, col) <= (others => '0');
                                        input_b(row, col) <= (others => '0');
                                    end if;
                                end if;
                            end if;
                        end loop;
                    end loop;

                    -----------------------------------------------------------------
                    -- COMPUTE_SOFTMAX: run max→taylor, then quantize outputs
                    -----------------------------------------------------------------
                when compute_softmax =>
                    if softmax_ready = '1' and softmax_ctr < 4 then
                        softmax_ctr <= softmax_ctr + 1;
                        for i in 0 to 3 loop
                            -- if softmax_out(i) > to_signed(2048, 64) then
                            --     bram_softmax(i + 4 * softmax_ctr) <= to_signed(2048, 11);
                            -- elsif softmax_out(i) < to_signed(-2048, 64) then
                            --     bram_softmax(i + 4 * softmax_ctr) <= to_signed(-204, 11);
                            -- else
                            bram_softmax(i + 4 * softmax_ctr) <= resize(softmax_out(i), 11);
                            -- end if;
                        end loop;
                    end if;

                    -- max/taylor handshake
                    if max_ctr < 5 then
                        if find_max = '1' then
                            taylor_calc <= '1';
                            max_ctr     <= max_ctr + 1;
                        elsif processing = '0' and max_ctr < 4 then
                            start_max <= '1';
                        else
                            start_max <= '0';
                        end if;

                        if taylor_calc = '1' then
                            if delay_ctr = 5 then
                                delay_ctr   <= 0;
                                taylor_calc <= '0';
                                taylor_done <= '1';
                            else
                                delay_ctr   <= delay_ctr + 1;
                                taylor_done <= '0';
                            end if;
                        elsif max_ctr = 4 and softmax_ready = '1' then
                            max_ctr     <= 0;
                            clear_mac   <= '1';
                            state       <= compute_attention;
                            taylor_done <= '0';
                        else
                            taylor_done <= '0';
                        end if;
                    else
                        taylor_done <= '0';
                    end if;

                    -----------------------------------------------------------------
                    -- COMPUTE_ATTENTION: weighted-sum of V
                    -----------------------------------------------------------------
                when compute_attention =>
                    taylor_done <= '0';
                    clear_mac   <= '0';
                    store_attn  <= '0';

                    if start_mac = '0' and done_mac = '1' then
                        clear_mac  <= '1';
                        store_attn <= '1';

                        if v_ext_iter = 21 and done_mac = '1'then
                            state <= idle;
                        else
                            v_ext_iter <= v_ext_iter + 1;
                        end if;

                    elsif start_mac = '0' and done_mac = '0' then
                        start_mac <= '1';

                    else
                        start_mac <= '0';
                    end if;

                    if v_ext_iter < 21 then
                        -- Matrix Mult Inputs => bram_softmax & bram_v [V0, V1, V2, V3]
                        for row in 0 to 11 loop
                            for col in 0 to 3 loop
                                k_idx   := row / 3;
                                r_mod_3 := row mod 3;

                                input_a(row, col) <= resize(signed(bram_softmax(4 * k_idx + col)), 8);
                                input_a(row, 4)   <= to_signed(0, 8);

                                input_b(row, col) <= resize(shift_right(
                                signed(bram_qkv(128 + 192 * col + r_mod_3 + 3 * v_ext_iter)), 7), 8);
                                input_b(row, 4) <= to_signed(0, 8);
                            end loop;
                        end loop;
                    else
                        -- Matrix Mult Inputs => bram_softmax & bram_v [V0, V1, V2, V3]
                        for row in 0 to 11 loop
                            for col in 0 to 3 loop
                                k_idx   := row / 3;
                                r_mod_3 := row mod 3;

                                if ((row = 0) or (row = 3) or (row = 6) or (row = 9)) then
                                    input_a(row, col) <= resize(signed(bram_softmax(4 * k_idx + col)), 8);
                                    input_a(row, 4)   <= to_signed(0, 8);

                                    input_b(row, col) <= resize(shift_right(
                                    signed(bram_qkv(128 + 192 * col + r_mod_3 + 3 * v_ext_iter)), 7), 8);
                                    input_b(row, 4) <= to_signed(0, 8);
                                else
                                    input_a(row, col) <= (others => '0');
                                    input_b(row, col) <= (others => '0');
                                end if;
                            end loop;
                        end loop;
                    end if;

                when others =>
                    state <= idle;
            end case;
        end if;
    end process;

    -------------------------------------------------------------------
    -- 4) MAX FINDER (unchanged)
    -------------------------------------------------------------------
    process (clk, rst)
        variable score_idx : integer;
    begin
        if rst = '1' then
            max_lut     <= (others => (others => '0'));
            max_val     <= (others => '0');
            current_max <= (others => '0');
            index       <= 0;
            find_max    <= '0';
            processing  <= '0';
            max_index   <= 0;
        elsif rising_edge(clk) then
            if start_max = '1' then
                current_max <= signed(bram_scores(4 * max_index));
                index       <= 1;
                find_max    <= '0';
                processing  <= '1';
            elsif processing = '1' then
                if index < 4 then
                    score_idx := index + 4 * max_index;
                    if score_idx <= 15 and signed(bram_scores(score_idx)) > current_max then
                        current_max  <= signed(bram_scores(score_idx));
                    end if;
                    index <= index + 1;
                else
                    max_val    <= current_max;
                    find_max   <= '1';
                    processing <= '0';
                end if;
            elsif find_max = '1' then
                max_lut(max_index) <= max_val;
                find_max           <= '0';
                if max_index < 3 then
                    max_index <= max_index + 1;
                else
                    max_index <= 0;
                end if;
            end if;
        end if;
    end process;

    -------------------------------------------------------------------
    -- 5) TAYLOR APPROX & SOFTMAX (unchanged)
    -------------------------------------------------------------------
    process (clk, rst)
    begin
        if rst = '1' then
            z_diff        <= (others => (others => '0'));
            z_diff_sq     <= (others => (others => '0'));
            z_diff_cu     <= (others => (others => '0'));
            term1         <= (others => (others => '0'));
            term2         <= (others => (others => '0'));
            term3         <= (others => (others => '0'));
            taylor_sum    <= (others => '0');
            softmax_ready <= '0';
        elsif rising_edge(clk) then
            for i in 0 to 3 loop
                if taylor_calc = '1' then
                    z_diff(i)        <= bram_scores(i + 4 * (max_ctr - 1)) - max_lut(max_ctr - 1);
                    z_diff_q15(i)    <= (resize(z_diff(i), 16));
                    z_diff_sq(i)     <= z_diff_q15(i) * z_diff_q15(i);
                    z_diff_cu(i)     <= z_diff_sq(i) * z_diff_q15(i);
                    term1(i)         <= shift_left(resize(z_diff_q15(i), 64), 30);
                    term2(i)         <= shift_left(shift_right(resize(z_diff_sq(i), 64), 1), 15);
                    term3(i)         <= shift_right(resize(z_diff_cu(i) * to_signed(21846, 64), 64), 0);
                    taylor_approx(i) <= (to_signed(1, 64)) + term1(i) + term2(i) + term3(i);
                end if;
            end loop;
            if taylor_done = '1' then
                softmax_ready <= '1';
                taylor_sum    <= resize(taylor_approx(0), 64)
                    + resize(taylor_approx(1), 64)
                    + resize(taylor_approx(2), 64)
                    + resize(taylor_approx(3), 64);
            else
                softmax_ready <= '0';
            end if;
        end if;
    end process;

    -- combine into final softmax
    process (taylor_approx, taylor_sum)
    begin
        for i in 0 to 3 loop
            if taylor_sum /= 0 then
                softmax_out(i) <= shift_left(to_signed(1, 64), 8) - abs((shift_left(taylor_approx(i), 8)) / taylor_sum);

            else
                softmax_out(i) <= (others => '0');
            end if;
        end loop;
    end process;

    -------------------------------------------------------------------
    -- 6) ATTENTION WRITEBACK (unchanged)
    -------------------------------------------------------------------
    fetch_attn <= store_attn; -- Trigger for FFN 

end architecture;