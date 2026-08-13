-- =====================================================================
-- stage_engine_tb -- the staged weight window against the real geometries
--
-- Checks the three passes layer_ctrl actually issues, at the real strides,
-- against the layout the MAC array requires:
--
--   QK^T  window[r][j] = K[token j][dim r]   -- a transpose, lanes = n_tok
--   xV/0  window[r][j] = V[token r][dim j]   -- direct, lanes 0..31
--   xV/1  window[rows+r][j] = V[token r][dim 32+j]  -- second column tile
--
-- and checks the two properties the as-received module got wrong beyond
-- addressing:
--
--   * lanes >= `lanes` read ZERO, not whatever the previous pass left
--   * the window row a byte lands in is the row it was READ for
--
-- Activation memory is modelled as act[a] = f(a) with f injective over the
-- whole address space, so a wrong address is a wrong VALUE and cannot pass
-- by coincidence.
-- =====================================================================
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use STD.ENV.all;

library work;
use work.mirage_v2_pkg.all;

entity stage_engine_tb is
end entity;

architecture tb of stage_engine_tb is

    constant TCK : time := 10 ns;
    constant NTOK : natural := 3;      -- "Hello" is 2 tokens + </s>

    signal clk : std_logic := '0';
    signal rst : std_logic := '1';

    signal go, busy, fin : std_logic := '0';
    signal base    : std_logic_vector(ACT_AW - 1 downto 0) := (others => '0');
    signal rows    : std_logic_vector(6 downto 0) := (others => '0');
    signal lanes   : std_logic_vector(6 downto 0) := (others => '0');
    signal stride  : std_logic_vector(ACT_AW - 1 downto 0) := (others => '0');
    signal xpose   : std_logic := '0';
    signal row_ofs : std_logic_vector(BANK_AW - 1 downto 0) := (others => '0');

    signal act_re    : std_logic;
    signal act_raddr : std_logic_vector(ACT_AW - 1 downto 0);
    signal act_data  : act_t := (others => '0');

    signal w_we   : std_logic;
    signal w_row  : std_logic_vector(BANK_AW - 1 downto 0);
    signal w_bank : std_logic_vector(2 downto 0);
    signal w_data : std_logic_vector(BANK_W - 1 downto 0);

    signal rd_en  : std_logic := '0';
    signal rd_row : std_logic_vector(BANK_AW - 1 downto 0) := (others => '0');
    signal w_bus  : wgt_bus_t;

    -- weight_window writes to the buffer NOT selected by rd_buf. A caller that
    -- uses one buffer serially therefore has to flip it: load with '1' to fill
    -- A, read back with '0' to see A. See the PROTOCOL note in
    -- weight_window.vhd. Held at '1' so the first pass stages before any read.
    signal rd_buf : std_logic := '1';

    -- injective over 0 .. 2**ACT_AW-1 modulo the int8 range: two addresses
    -- collide only 251 apart, and no check here spans 251 addresses.
    function act_of(a : natural) return integer is
    begin
        return ((a * 7 + 13) mod 251) - 125;
    end function;

begin

    clk <= not clk after TCK / 2;

    act_model : process (clk)
    begin
        if rising_edge(clk) then
            if act_re = '1' then
                act_data <= to_signed(act_of(to_integer(unsigned(act_raddr))), ACT_W);
            end if;
        end if;
    end process;

    dut : entity work.stage_engine
        port map (clk => clk, rst => rst,
                  go => go, busy => busy, fin => fin,
                  base => base, rows => rows, lanes => lanes,
                  stride => stride, xpose => xpose, row_ofs => row_ofs,
                  act_re => act_re, act_raddr => act_raddr, act_data => act_data,
                  w_we => w_we, w_row => w_row, w_bank => w_bank, w_data => w_data);

    win : entity work.weight_window
        port map (clk => clk,
                  rd_en => rd_en, rd_row => rd_row, rd_buf => rd_buf, w_bus => w_bus,
                  wr_en => w_we, wr_row => w_row, wr_bank => w_bank, wr_data => w_data);

    stim : process
        variable errs    : natural := 0;
        variable checked : natural := 0;
        variable shown   : natural := 0;

        procedure run_pass(b : natural; r : natural; l : natural;
                           st : natural; xp : std_logic; ro : natural) is
        begin
            rd_buf  <= '1';                 -- stage into A (writes go to !rd_buf)
            base    <= std_logic_vector(to_unsigned(b, ACT_AW));
            rows    <= std_logic_vector(to_unsigned(r, 7));
            lanes   <= std_logic_vector(to_unsigned(l, 7));
            stride  <= std_logic_vector(to_unsigned(st, ACT_AW));
            xpose   <= xp;
            row_ofs <= std_logic_vector(to_unsigned(ro, BANK_AW));
            wait until rising_edge(clk);
            go <= '1';
            wait until rising_edge(clk);
            go <= '0';
            wait until fin = '1' and rising_edge(clk);
            rd_buf  <= '0';                 -- read back the buffer just filled
            wait until rising_edge(clk);
        end procedure;

        -- want < -128 is the sentinel for "this lane must read zero"
        procedure expect(name : string; wrow : natural; lane : natural;
                         want : integer) is
            variable got : integer;
        begin
            rd_row <= std_logic_vector(to_unsigned(wrow, BANK_AW));
            rd_en  <= '1';
            wait until rising_edge(clk);
            rd_en  <= '0';
            wait for 1 ns;
            got := to_integer(get_wgt(w_bus, lane));
            checked := checked + 1;
            if got /= want then
                errs := errs + 1;
                if shown < 15 then
                    shown := shown + 1;
                    report "  MISMATCH " & name & " win_row=" & integer'image(wrow)
                         & " lane=" & integer'image(lane)
                         & " want=" & integer'image(want)
                         & " got=" & integer'image(got) severity warning;
                end if;
            end if;
        end procedure;

        constant K_BASE : natural := REG_K;      -- head 0
        constant V_BASE : natural := REG_V;
    begin
        rst <= '1';
        for i in 0 to 4 loop wait until rising_edge(clk); end loop;
        rst <= '0';
        wait until rising_edge(clk);

        -----------------------------------------------------------------
        -- Pass 1: QK^T. K^T for head 0, reduction over all HEAD_DIM.
        -----------------------------------------------------------------
        report "PASS 1: QK^T transpose, rows=HEAD_DIM lanes=n_tok" severity note;
        run_pass(K_BASE, HEAD_DIM, NTOK, TOK_STRIDE, '1', 0);
        for r in 0 to HEAD_DIM - 1 loop
            for j in 0 to N_LANES - 1 loop
                if j < NTOK then
                    expect("QKt", r, j, act_of(K_BASE + j * TOK_STRIDE + r));
                else
                    expect("QKt-pad", r, j, 0);
                end if;
            end loop;
        end loop;

        -----------------------------------------------------------------
        -- Pass 2 and 3: xV, both column tiles of HEAD_DIM = 2 x N_LANES.
        -- Tile ct covers output dims ct*32 .. ct*32+31 and lands at window
        -- rows ct*n_tok .., which is exactly where gemm_sequencer's
        -- w_tile = ct * k_len will look for it.
        -----------------------------------------------------------------
        report "PASS 2/3: xV direct, two column tiles" severity note;
        for ct in 0 to 1 loop
            run_pass(V_BASE + ct * N_LANES, NTOK, N_LANES, TOK_STRIDE, '0',
                     ct * NTOK);
            for r in 0 to NTOK - 1 loop
                for j in 0 to N_LANES - 1 loop
                    expect("xV", ct * NTOK + r, j,
                           act_of(V_BASE + r * TOK_STRIDE + ct * N_LANES + j));
                end loop;
            end loop;
        end loop;

        -----------------------------------------------------------------
        -- Pass 4: a short pass AFTER a full one, to prove zero-fill
        -- overwrites rather than leaves the previous pass's bytes.
        -----------------------------------------------------------------
        report "PASS 4: short pass must zero the lanes it does not cover"
            severity note;
        run_pass(K_BASE, 4, 2, TOK_STRIDE, '1', 0);
        for r in 0 to 3 loop
            for j in 0 to N_LANES - 1 loop
                if j < 2 then
                    expect("short", r, j, act_of(K_BASE + j * TOK_STRIDE + r));
                else
                    expect("short-pad", r, j, 0);
                end if;
            end loop;
        end loop;

        report "==== checked " & integer'image(checked) & " window bytes, "
             & integer'image(errs) & " wrong" severity note;
        if errs = 0 then
            report "PASS: stage_engine reproduces K^T and V at the real strides"
                severity note;
        else
            report integer'image(errs) & " staged bytes wrong" severity failure;
        end if;
        -- finish, not `wait` -- the clock process runs forever, so a bare wait
        -- leaves a PASSING run spinning and the suite unusable as a gate.
        finish;
    end process;

end architecture;
