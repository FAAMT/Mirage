-- =====================================================================
-- mirage_axi_wrap.vhd -- VHDL-93 shim so the block design can consume
-- mirage_axi_top
--
-- WHY THIS FILE EXISTS
--
-- Vivado's two routes into a block design both object to a VHDL-2008
-- top, in different ways:
--
--   create_bd_cell -type module
--     [filemgmt 56-195] Reference 'mirage_axi_top' contains top file
--     'mirage_axi_top.vhd' of type VHDL 2008. This type is not allowed
--     as the top file in the reference.
--     -- a hard refusal.
--
--   ipx::package_project
--     [IP_Flow 19-5655] Packaging a component with a VHDL-2008 top file
--     is not fully supported.
--     -- a critical warning, i.e. it proceeds into territory Xilinx
--     declines to stand behind.
--
-- The restriction is on the TOP FILE only. Everything below it stays
-- VHDL-2008 and stays exactly as verified; this shim is plain 93 and
-- does nothing but rename the boundary, so both routes open up and the
-- 19-5655 warning goes away.
--
-- WHAT IT MUST NOT DO
--
-- No logic. No renaming. No width arithmetic, no package references --
-- every width here is a literal, because a package constant in a port
-- declaration is what broke IP packaging in the first place
-- ([IP_Flow 19-627] on s_axis_tdata). Port names are character-for-
-- character those of mirage_axi_top, because Vivado infers the AXI4-Lite
-- and AXI-Stream interfaces from the s_axi_* / s_axis_* prefixes and
-- renaming anything here would silently break that inference.
--
-- A port map is the entire contents. If this file ever needs a thought
-- in it, the thought belongs in mirage_axi_top.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;

entity mirage_axi_wrap is
    port
    (
        aclk    : in std_logic;
        aresetn : in std_logic;

        -- AXI4-Lite control
        s_axi_awaddr  : in  std_logic_vector(7 downto 0);
        s_axi_awvalid : in  std_logic;
        s_axi_awready : out std_logic;
        s_axi_wdata   : in  std_logic_vector(31 downto 0);
        s_axi_wstrb   : in  std_logic_vector(3 downto 0);
        s_axi_wvalid  : in  std_logic;
        s_axi_wready  : out std_logic;
        s_axi_bresp   : out std_logic_vector(1 downto 0);
        s_axi_bvalid  : out std_logic;
        s_axi_bready  : in  std_logic;
        s_axi_araddr  : in  std_logic_vector(7 downto 0);
        s_axi_arvalid : in  std_logic;
        s_axi_arready : out std_logic;
        s_axi_rdata   : out std_logic_vector(31 downto 0);
        s_axi_rresp   : out std_logic_vector(1 downto 0);
        s_axi_rvalid  : out std_logic;
        s_axi_rready  : in  std_logic;

        -- AXI4-Stream weight input
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;
        s_axis_tdata  : in  std_logic_vector(31 downto 0);
        s_axis_tlast  : in  std_logic;

        irq : out std_logic
    );
end entity;

architecture rtl of mirage_axi_wrap is
begin

    u_top : entity work.mirage_axi_top
        port map (
            aclk    => aclk,
            aresetn => aresetn,

            s_axi_awaddr  => s_axi_awaddr,
            s_axi_awvalid => s_axi_awvalid,
            s_axi_awready => s_axi_awready,
            s_axi_wdata   => s_axi_wdata,
            s_axi_wstrb   => s_axi_wstrb,
            s_axi_wvalid  => s_axi_wvalid,
            s_axi_wready  => s_axi_wready,
            s_axi_bresp   => s_axi_bresp,
            s_axi_bvalid  => s_axi_bvalid,
            s_axi_bready  => s_axi_bready,
            s_axi_araddr  => s_axi_araddr,
            s_axi_arvalid => s_axi_arvalid,
            s_axi_arready => s_axi_arready,
            s_axi_rdata   => s_axi_rdata,
            s_axi_rresp   => s_axi_rresp,
            s_axi_rvalid  => s_axi_rvalid,
            s_axi_rready  => s_axi_rready,

            s_axis_tvalid => s_axis_tvalid,
            s_axis_tready => s_axis_tready,
            s_axis_tdata  => s_axis_tdata,
            s_axis_tlast  => s_axis_tlast,

            irq => irq
        );

end architecture;
