

# Fetch historical data for multiple stocks
bse_largecap = ["ABB.BO","ADANIENG.BO","ADANIENT.BO","ADANIGREEN.BO","ADANIPORTS.BO","ADANIPOWER.BO","ATGL.NS","AWL.NS","AMBUJACEM.BO","APOLLOHOSP.BO","ASIANPAINT.BO","DMART.BO","AXISBANK.BO","BAJAJ-AUTO.BO","BAJFINANCE.BO","BAJAJFINSV.BO","BAJAJHLDNG.BO","BANDHANBNK.BO","BANKBARODA.BO","BERGEPAINT.BO","BEL.BO","BPCL.BO","BHARTIARTL.BO","BOSCHLTD.BO","BRITANNIA.BO","CHOLAFIN.BO","CIPLA.BO","COALINDIA.BO","DABUR.BO","DIVISLAB.BO","DLF.BO","DRREDDY.BO","EICHERMOT.BO","NYKAA.NS","GAIL.BO","GODREJCP.BO","GRASIM.BO","HAVELLS.BO","HCLTECH.BO","HDFCAMC.BO","HDFCBANK.BO","HDFCLIFE.BO","HEROMOTOCO.BO","HINDALCO.BO","HAL.BO","HINDUNILVR.BO","HINDZINC.BO","ICICIBANK.BO","ICICIGI.BO","ICICIPRULI.BO","IOC.BO","INDUSTOWER.BO","INDUSINDBK.BO","NAUKRI.BO","INFY.BO","INDIGO.BO","ITC.BO","JIOFIN.NS","JSWSTEEL.BO","KOTAKBANK.BO","LT.BO","LICI.NS","MINDTREE.BO","M&M.BO","MANKIND.NS","MARICO.BO","MARUTI.BO","NESTLEIND.BO","NTPC.BO","ONGC.BO","PAYTM.NS","PIDILITIND.BO","POWERGRID.BO","PNB.BO","RELIANCE.BO","SBICARD.BO","SBILIFE.BO","SHREECEM.BO","SIEMENS.BO","SRF.BO","SBIN.BO","SUNPHARMA.BO","TCS.BO","TATACONSUM.BO","TATAMOTORS.BO","TATAMTRDVR.BO","TATAPOWER.BO","TATASTEEL.BO","TECHM.BO","TITAN.BO","ULTRACEMCO.BO","MCDOWELL-N.BO","UPL.BO","VBL.BO","VEDL.BO","WIPRO.BO","ZOMATO.BO","ZYDUSLIFE.NS"]
bse_midcap = ["3MINDIA.BO","AARTIIND.BO","ABBOTINDIA.BO","ACC.BO","ABCAPITAL.BO","ABFRL.BO","AJANTPHARM.BO","ALKEM.BO","APLAPOLLO.BO","ASHOKLEY.BO","ASTRAL.BO","AUBANK.BO","AUROPHARMA.BO","BALKRISIND.BO","BANKINDIA.BO","BAYERCROP.BO","BHARATFORG.BO","BHEL.BO","BIOCON.BO","CANBK.BO","CASTROLIND.BO","CGPOWER.BO","CLEANSCIENCE.BO","COLPAL.BO","CONCOR.BO","COROMANDEL.BO","CRISIL.BO","CROMPTON.BO","CUMMINSIND.BO","DALBHARAT.BO","DEEPAKNTR.BO","DELHIVERY.NS","EMAMILTD.BO","ENDURANCE.BO","EXIDEIND.BO","FEDERALBNK.BO","GICRE.BO","GILLETTE.BO","GLAND.BO","GLAXO.BO","GLENMARK.BO","GMRINFRA.BO","GODREJIND.BO","GODREJPROP.BO","GUJFLUORO.BO","GUJGASLTD.BO","HINDPETRO.BO","HONAUT.BO","ISEC.BO","IDBI.BO","IDFCFIRSTB.BO","INDIANB.BO","INDHOTEL.BO","IOB.BO","IRCTC.BO","IRFC.BO","IREDAL.BO","IGL.BO","IPCALAB.BO","JINDALSTEL.BO","JSWENERGY.BO","JSWINFRA.NS","JUBLFOOD.BO","KANSAINER.BO","L&TFH.BO","LTTS.BO","LAURUSLABS.BO","LICHSGFIN.BO","LINDEINDIA.BO","LUPIN.BO","LODHA.BO","M&MFIN.BO","MFSL.BO","MAXHEALTH.BO","MPHASIS.BO","MRF.BO","MUTHOOTFIN.BO","NHPC.BO","NAM-INDIA.BO","NMDC.BO","NUVOCO.BO","OBEROIRLTY.BO","OIL.BO","OFSS.BO","PAGEIND.BO","PATANJALI.NS","PAYTM.BO","PERSISTENT.BO","PETRONET.BO","PIIND.BO","PEL.BO","POLYCAB.BO","PFC.BO","PGHH.BO","RAJESHEXPO.BO","RAMCOCEM.BO","RECLTD.BO","RELAXO.BO","SAMARTI.BO","SCHAEFFLER.BO","SRTRANSFIN.BO","SJVN.BO","SOLARINDS.BO","SONACOMS.BO","STARHEALTH.NS","SAIL.BO","SUNTV.BO","SUPREMEIND.BO","TATACOMM.BO","TATAELXSI.BO","TATATECH.BO","NIACL.BO","TORNTPHARM.BO","TORNTPOWER.BO","TRENT.BO","TIINDIA.BO","TVSMOTOR.BO","UCOBANK.BO","UNIONBANK.BO","UBL.BO","UNOMINDA.BO","VEDL.BO","IDEA.BO","VOLTAS.BO","WHIRLPOOL.BO","YESBANK.BO","ZEEL.BO"]
bse_smallcap = ["360ONE.NS","3IINFOTECH.NS","5PAISA.NS","63MOONS.NS","AARTIDRUGS.NS","AARTIPHARM.NS","AAVAS.NS","AHL.NS","ACCELYA.NS","ACE.NS","ADFFOODS.NS","ABSLAMC.NS","AVL.BO","ADORWELD.NS","ADVENZYMES.NS","AEGISCHEM.NS","AEROFLEX.NS","AETHER.NS","AFFLE.NS","AGARIND.NS","AGIGREEN.NS","ATFL.NS","AGSTRA.NS","AHLUCONT.NS","AIAENG.NS","AJMERA.NS","AKZOINDIA.NS","ALEMBICLTD.NS","APLLTD.NS","ALICON.NS","ALKYLAMINE.NS","ACLGATI.NS","ALLCARGO.NS","ATL.NS","ALLSEC.NS","ALOKINDS.NS","AMARAJABAT.NS","AMBER.NS","AMBIKCO.NS","AMIORG.NS","ANANDRATHI.NS","ANANTRAJ.NS","ANDHRAPAP.NS","ANDHRAPET.NS","ANDREWYU.NS","ANGELONE.NS","AWHCL.NS","ANURAS.NS","APARINDS.NS","APCOTEXIND.NS","APOLLO.NS","APOLLOPIPE.NS","APOLLOTYRE.NS","APTECHT.NS","APTUS.NS","ARCHIECHEM.NS","ARIHANTCAP.NS","ARIHANTSUP.NS","ARMANFIN.NS","ARTEMISMED.NS","ARVINDFASN.NS","ARVIND.NS","ARVSMART.NS","ASAHIINDIA.NS","ASHAPURMIN.NS","ASHIANA.NS","ASHOKA.NS","ASIANTILES.NS","ASKAUTOLTD.NS","ASALCBR.NS","ASTEC.NS","ASTERDM.NS","ASTRAMICRO.NS","ASTRAZEN.NS","ATULAUTO.NS","ATUL.NS","AURIONPRO.NS","AIIL.BO","AUTOAXLES.NS","ASAL.NS","AVADHSUGAR.NS","AVALON.NS","AVANTEL.NS","AVANTIFEED.NS","AVTNPL.NS","AXISCADES.NS","AZAD.NS","BLKASHYAP.NS","BAJAJCON.NS","BAJAJELEC.NS","BAJAJHCARE.NS","BAJAJHIND.NS","BALAMINES.NS","BALMLAWRIE.NS","BLIL.BO","BALRAMCHIN.NS","BALUFORGE.BO","BANCOINDIA.NS","MAHABANK.NS","BANARISUG.NS","BARBEQUE.NS","BASF.NS","BATAINDIA.NS","BCLIND.NS","BLAL.NS","BEML.NS","BESTAGRO.NS","BFINVEST.NS","BFUTILITIE.NS","BHAGERIA.NS","BHAGCHEM.NS","BEPL.NS","BHARATBIJ.NS","BDL.NS","BHARATWIRE.NS","BIGBLOC.NS","BIKAJI.NS","BIRLACORPN.NS","BSOFT.NS","BLACKBOX.NS","BLACKROSE.NS","BLISSGVS.NS","BLS.NS","BLUEDART.NS","BLUEJET.NS","BLUESTARCO.NS","BODALCHEM.NS","BOMDYEING.NS","BOROLTD.NS","BORORENEW.NS","BRIGADE.NS","BUTTERFLY.NS","CEINFO.NS","CAMLINFINE.NS","CAMPUS.NS","CANFINHOME.NS","CANTABIL.NS","CAPACITE.NS","CAPLIPOINT.NS","CGCL.NS","CARBORUNIV.NS","CARERATING.NS","CARTRADE.NS","CARYSIL.NS","CCL.NS","CEATLTD.NS","CELLO.NS","CENTRALBK.NS","CENTRUM.NS","CENTUM.NS","CENTENKA.NS","CENTURYPLY.NS","CENTURYTEX.NS","CERA.NS","CESC.NS","CHALET.NS","CLSEL.NS","CHAMBLFERT.NS","CHEMCON.NS","CHEMPLASTS.NS","CHENNPETRO.NS","CHEVIOT.NS","CHOICEIN.NS","CHOLAHOLD.NS","CIEINDIA.NS","CIGNITITEC.NS","CUB.NS","CMSINFO.NS","COCHINSHIP.NS","COFFEEDAY.NS","COFORGE.NS","CAMS.NS","CONCORDBIO.NS","CONFIPET.NS","CONTROLPR.NS","COSMOFIRST.NS","CRAFTSMAN.NS","CREDITACC.NS","MUFTI.BO","CRESSANDA.NS","CSBBANK.NS","CYIENTDLM.NS","CYIENT.NS","DLINKINDIA.NS","DALMIASUG.NS","DATAPATTNS.NS","DATAMATICS.NS","DBCORP.NS","DCBBANK.NS","DCMSHRIRAM.NS","DCW.NS","DCXINDIA.NS","DDEVPLASTIK.BO","DECCANCE.NS","DEEPINDS.NS","DEEPAKFERT.NS","DELTACORP.NS","DEN.NS","DEVYANI.NS","DHAMPURBIO.NS","DHAMPURSUG.NS","DHANI.NS","DHANUKA.NS","DHARAMSI.NS","DCGL.NS","DHUNINV.NS","DBL.NS","DISHTV.NS","DISHMAN.NS","DIVGI.NS","DIXON.NS","DODLA.NS","DOLLAR.NS","DOMS.NS","LALPATHLAB.NS","DREAMFOLKS.NS","DREDGECORP.NS","DWARKESH.NS","DYNAMATECH.NS","EASEMYTRIP.NS","ECLERX.NS","EDELWEISS.NS","EIDPARRY.NS","EIHAHOTELS.NS","EIHOTEL.NS","EKIEGS.NS","ELECON.NS","EMIL.NS","ELECTCAST.NS","ELGIEQUIP.NS","ELIN.NS","ELPROINTL.NS","EMAMIPAP.NS","EMSLIMITED.NS","EMUDHRA.NS","ENGINERSIN.NS","ENIL.NS","EPIGRAL.NS","EPL.NS","EQUITASBNK.NS","ERIS.NS","ESABINDIA.NS","ESAFSFB.NS","ESCORTS.NS","ESTER.NS","ETHOSLTD.NS","EUREKAFORBE.BO","EVEREADY.NS","EVERESTIND.NS","EKC.NS","EXCELINDUS.NS","EXPLEOSOL.NS","FAIRCHEM.NS","FAZETHREE.NS","FDC.NS","FEDFINA.NS","FMGOETZE.NS","FIEMIND.NS","FILATEX.NS","FINEORG.NS","FCL.NS","FINOPB.NS","FINCABLES.NS","FINPIPE.NS","FSL.NS","FIVESTAR.NS","FLAIR.NS","FOODSIN.NS","FORCEMOT.NS","FORTIS.NS","FOSECOIND.NS","FUSION.NS","GMBREW.NS","GNA.NS","GRINFRA.NS","GABRIEL.NS","GALAXYSURF.NS","GALLANTT.NS","GANDHAR.NS","GANDHITUBE.NS","GANESHBENZ.NS","GANESHHOUC.NS","GANECOS.NS","GRSE.NS","GARFILA.NS","GARFIBRES.NS","GDL.NS","GEPIL.NS","GET&D.NS","GENESYS.NS","GENSOL.NS","GENUSPOWER.NS","GEOJITFSL.NS","GFLLIMITED.NS","GHCL.NS","GHCLTEXTIL.NS","GICHSGFIN.NS","GLENMARK.NS","MEDANTA.NS","GSURF.NS","GLOBUSSPR.NS","GMMPFAUDLR.NS","GMRINFRA.NS","GOFASHION.NS","GOCLCORP.NS","GPIL.NS","GODFRYPHLP.NS","GODREJAGRO.NS","GOKEX.NS","GOKUL.NS","GOLDIAM.NS","GOODLUCK.NS","GOODYEAR.NS","GRANULES.NS","GRAPHITE.NS","GRAUWEIL.NS","GRAVITA.NS","GESHIP.NS","GREAVESCOT.NS","GREENLAM.NS","GREENPANEL.NS","GREENPLY.NS","GRINDWELL.NS","GRMOVER.NS","GTLINFRA.NS","GTPL.NS","GUFICBIO.NS","GUJALKALI.NS","GAEL.NS","GIPCL.NS","GMDCLTD.NS","GNFC.NS","GPPL.NS","GSFC.NS","GSPL.NS","GUJTHEMIS.NS","GULFOILLUB.NS","GULPOLY.NS","HGINFRA.NS","HAPPSTMNDS.NS","HAPPYFORGE.NS","HARDWYN.NS","HARIOMPIPE.NS","HARSHAENG.NS","HATHWAY.NS","HATSUN.NS","HAWKINCOOK.BO","HBLPOWER.NS","HCG.NS","HEG.NS","HEIDELBERG.NS","HEMIPROP.NS","HERANBA.NS","HERCULES.NS","HERITGFOOD.NS","HESTERBIO.NS","HEUBACHIND.NS","HFCL.NS","HITECH.NS","HIKAL.NS","HIL.NS","HSCL.NS","HIMATSEIDE.NS","HGS.NS","HCC.NS","HINDCOPPER.NS","HINDUSTAN.NS","HINDOILEXP.NS","HINDWARE.NS","POWERINDIA.NS","HLEGLAS.NS","HOTELLEELA.NS","HMAAGRO.NS","HOMEFIRST.NS","HONASA.NS","HONDAPOWER.NS","HUDCO.NS","HPAL.NS","HPL.NS","HUHTAMAKI.NS","ICRA.NS","IDEA.NS","IDFC.NS","IFBIND.NS","IFCI.NS","IFGLEXPOR.NS","IGPL.NS","IGARASHI.NS","IIFL.NS","IIFLSEC.NS","IKIO.NS","IMAGICAA.NS","INDIACEM.NS","INDIAGLYCO.NS","INDNIPPON.NS","IPL.NS","INDIASHLTR.NS","ITDC.NS","IBULHSGFIN.NS","IBREALEST.NS","INDIAMART.NS","IEX.NS","INDIANHUME.NS","IMFA.NS","INDIGOPNTS.NS","INDOAMIN.NS","ICIL.NS","INDORAMA.NS","INDOCO.NS","INDREMEDI.NS","INFIBEAM.NS","INFOBEAN.NS","INGERRAND.NS","INNOVACAP.NS","INOXGREEN.NS","INOXINDIA.NS","INOXWIND.NS","INSECTICID.NS","INTELLECT.NS","IOLCP.NS","IONEXCHANG.NS","IRB.NS","IRCON.NS","IRMENERGY.NS","ISGEC.NS","ITDCEM.NS","ITI.NS","JKUMAR.NS","JKTYRE.NS","JBCHEPHARM.NS","JKCEMENT.NS","JAGRAN.NS","JAGSNPHARM.NS","JAIBALAJI.NS","JAICORPLTD.NS","JISLJALEQS.NS","JPASSOCIAT.NS","JPPOWER.NS","J&KBANK.NS","JAMNAAUTO.NS","JAYBARMARU.NS","JAYAGROGN.NS","JAYNECOIND.NS","JBMA.NS","JINDRILL.NS","JINDALPOLY.NS","JINDALSAW.NS","JSL.NS","JINDWORLD.NS","JKLAKSHMI.NS","JKPAPER.NS","JMFINANCIL.NS","JCHAC.NS","JSWHL.NS","JTEKTINDIA.NS","JTLIND.NS","JUBLINDS.NS","JUBLINGREA.NS","JUBLPHARMA.NS","JLHL.NS","JWL.NS","JUSTDIAL.NS","JYOTHYLAB.NS","JYOTIRES.BO","KABRAEXTRU.NS","KAJARIACER.NS","KALPATPOWR.NS","KALYANKJIL.NS","KALYANIFRG.NS","KSL.NS","KAMAHOLD.NS","KAMDHENU.NS","KAMOPAINTS.NS","KANORICHEM.NS","KTKBANK.NS","KSCL.NS","KAYNES.NS","KDDL.NS","KEC.NS","KEI.NS","KELLTONTEC.NS","KENNAMET.NS","KESORAMIND.NS","KKCL.NS","RUSTOMJEE.NS","KFINTECH.NS","KHAICHEM.NS","KINGFA.NS","KIOCL.NS","KIRIINDUS.NS","KIRLOSBROS.NS","KIRLFER.NS","KIRLOSIND.NS","KIRLOSENG.NS","KIRLPNU.NS","KITEX.NS","KMCSHIL.BO","KNRCON.NS","KOKUYOCMLN.NS","KOLTEPATIL.NS","KOPRAN.NS","KOVAI.NS","KPIGREEN.NS","KPITTECH.NS","KPRMILL.NS","KRBL.NS","KIMS.NS","KRSNAA.NS","KSB.NS","KSOLVES.NS","KUANTUM.NS","LAOPALA.NS","LAXMIMACH.NS","LANCER.NS","LANDMARK.NS","LATENTVIEW.NS","LXCHEM.NS","LEMONTREE.NS","LGBBROSLTD.NS","LIKHITHA.NS","LINCPEN.NS","LINCOLN.NS","LLOYDSENGG.NS","LLOYDSENT.BO","LLOYDSME.NS","DAAWAT.NS","LUMAXTECH.NS","LUMAXIND.NS","MMFL.NS","MKPL.NS","MCLOUD.BO","MGL.NS","MTNL.NS","MAHSCOOTER.NS","MAHSEAMLES.NS","MHRIL.NS","MAHLIFE.NS","MAHLOG.NS","MANINDS.NS","MANINFRA.NS","MANGCHEFER.NS","MANAKSIA.NS","MANALIPETC.NS","MANAPPURAM.NS","MANGLMCEM.NS","MRPL.NS","MVGJL.NS","MANORAMA.NS","MARATHON.NS","MARKSANS.NS","MASFIN.NS","MASTEK.NS","MATRIMONY.NS","MAXESTATE.NS","MAYURUNIQ.NS","MAZDOCK.NS","MEDICAMEQ.NS","MEDPLUS.NS","MEGH.NS","MENONBE.NS","METROBRAND.NS","METROPOLIS.NS","MINDACORP.NS","MIRZAINT.NS","MIDHANI.NS","MISHTANN.NS","MMTC.NS","MOIL.NS","MOLDTKPAC.NS","MONARCH.NS","MONTECARLO.NS","MOREPENLAB.NS","MOSCHIP.NS","MSUMI.NS","MOTILALOFS.NS","MPSLTD.NS","BECTORFOOD.NS","MSTCLTD.NS","MTARTECH.NS","MUKANDLTD.NS","MCX.NS","MUTHOOTMF.NS","NACLIND.NS","NAGAFERT.NS","NAHARINV.NS","NAHARSPING.NS","NSIL.NS","NH.NS","NATCOPHARM.NS","NATIONALUM.NS","NFL.NS","NAVINFLUOR.NS","NAVKARCORP.NS","NAVNETEDUL.NS","NAZARA.NS","NBCC.NS","NCC.NS","NCLIND.NS","NELCAST.NS","NELCO.NS","NEOGEN.NS","NESCO.NS","NETWEB.NS","NETWORK18.NS","NEULANDLAB.NS","NDTV.NS","NEWGEN.NS","NGLFINE.NS","NIITMTS.NS","NIITLTD.NS","NIKHILADH.NS","NILKAMAL.NS","NITINSPIN.NS","NITTAGELA.BO","NLCINDIA.NS","NSLNISP.NS","NOCIL.NS","NOVARTIND.NS","NRBBEARING.NS","NUCLEUS.NS","NUVAMA.NS","OLECTRA.NS","OMAXE.NS","ONMOBILE.NS","ONWARDTEC.NS","OPTIEMUS.NS","ORIENTBELL.NS","ORIENTCEM.NS","ORIENTELEC.NS","GREENPOWER.NS","ORIENTPPR.NS","OAL.NS","OCCL.NS","ORIENTHOT.NS","OSWALGREEN.NS","PAISALO.NS","PANACEABIO.NS","PANAMAPET.NS","PARADEEP.NS","PARAGMILK.NS","PARA.NS","PARAS.NS","PATELENG.NS","PAUSHAKLTD.NS","PCJEWELLER.NS","PCBL.NS","PDSL.NS","PGIL.NS","PENIND.NS","PERMAGNET.NS","PFIZER.NS","PGEL.NS","PHOENIXLTD.NS","PILANIINVS.NS","PPLPHARMA.NS","PITTILAM.NS","PIXTRANS.NS","PNBGILTS.NS","PNBHOUSING.NS","PNCINFRA.NS","POKARNA.NS","POLYMED.NS","POLYPLEX.NS","POONAWALLA.NS","POWERMECH.NS","PRAJIND.NS","PRAKASHSTL.NS","DIAMONDYD.NS","PRAVEG.NS","PRECAM.NS","PRECWIRE.NS","PRESTIGE.NS","PRICOLLTD.NS","PFOCUS.NS","PRIMO.NS","PRINCEPIPE.NS","PRSMJOHNSN.NS","PRIVISCL.NS","PGHL.NS","PROTEAN.BO","PRUDENT.NS","PSPPROJECT.NS","PFS.NS","PTC.NS","PTCIL.BO","PSB.NS","PUNJABCHEM.NS","PURVA.NS","PVRINOX.NS","QUESS.NS","QUICKHEAL.NS","QUINT.NS","RRKABEL.NS","RSYSTEMS.NS","RACLGEAR.NS","RADIANT.NS","RADICO.NS","RAGHAVPRO.NS","RVNL.NS","RAILTEL.NS","RAIN.NS","RAINBOW.NS","RAJRATAN.NS","RALLIS.NS","RRWL.NS","RAMASTEEL.NS","RAMCOIND.NS","RAMCOSYS.NS","RKFORGE.NS","RAMKY.NS","RANEHOLDIN.NS","RML.NS","RCF.NS","RATEGAIN.NS","RATNAMANI.NS","RTNINDIA.NS","RTNPOWER.NS","RAYMOND.NS","RBLBANK.NS","REDINGTON.NS","REDTAPE.NS","REFEX.NS","RIIL.NS","RELINFRA.NS","RPOWER.NS","RELIGARE.NS","RENAISSANCE.NS","REPCOHOME.NS","REPRO.NS","RESPONIND.NS","RBA.NS","RHIM.NS","RICOAUTO.NS","RISHABHINS.NS","RITES.NS","ROLEXRINGS.NS","ROSSARI.NS","ROSSELLIND.NS","ROTOPUMPS.NS","ROUTE.NS","ROHL.NS","RPGLIFE.NS","RPSGVENT.NS","RSWM.NS","RUBYMILLS.NS","RUPA.NS","RUSHIL.NS","SCHAND.NS","SHK.NS","SJS.NS","SPAL.NS","SADHANANIQ.NS","SAFARI.NS","SAGCEM.NS","SAILKALAM.NS","SALASAR.NS","SAMHIHOTEL.NS","SANDHAR.NS","SANDUMA.NS","SANGAMIND.NS","SANGHIIND.NS","SANGHVIMOV.NS","SANMITINFRA.NS","SANOFI.NS","SANSERA.NS","SAPPHIRE.NS","SARDAEN.NS","SAREGAMA.NS","SASKEN.NS","SASTASUNDR.NS","SATINDLTD.NS","SATIA.NS","SATIN.NS","SOTL.NS","SBFC.NS","SCHNEIDER.NS","SEAMECLTD.NS","SENCO.NS","SEPC.NS","SEQUENT.NS","SESHAPAPER.NS","SGFIN.NS","SHAILY.NS","SHAKTIPUMP.NS","SHALBY.NS","SHALPAINTS.NS","SHANKARA.NS","SHANTIGEAR.NS","SHARDACROP.NS","SHARDAMOTR.NS","SHAREINDIA.NS","SFL.NS","SHILPAMED.NS","SCI.NS","SHIVACEM.NS","SBCL.NS","SHIVALIK.NS","SHOPERSTOP.NS","SHREDIGCEM.NS","SHREEPUSHK.NS","RENUKA.NS","SHREYAS.NS","SHRIRAMPPS.NS","SHYAMMETL.NS","SIGACHI.NS","SIGNA.NS","SIRCA.NS","SIS.NS","SIYSIL.NS","SKFINDIA.NS","SKIPPER.NS","SMCGLOBAL.NS","SMLISUZU.NS","SMSPHARMA.NS","SNOWMAN.NS","SOBHA.NS","SOLARA.NS","SDBL.NS","SOMANYCERA.NS","SONATSOFTW.NS","SOUTHBANK.NS","SPANDANA.NS","SPECIALITY.NS","SPENCERS.NS","SPICEJET.NS","SPORTKING.NS","SRHHYPOLTD.NS","STGOBANQ.NS","STARCEMENT.NS","STEELXIND.NS","SSWL.NS","STEELCAS.NS","SWSOLAR.NS","STERTOOLS.NS","STRTECH.NS","STOVEKRAFT.NS","STAR.NS","STYLAMIND.NS","STYRENIX.NS","SUBEXLTD.NS","SUBROS.NS","SUDARSCHEM.NS","SUKHJITS.NS","SULA.NS","SUMICHEM.NS","SUMMITSEC.NS","SPARC.NS","SUNCLAYLTD.NS","SUNDRMFAST.NS","SUNFLAG.NS","SUNTECK.NS","SUPRAJIT.NS","SUPPETRO.NS","SUPRIYA.NS","SURAJEST.NS","SURYAROSNI.NS","SURYODAY.NS","SUTLEJTEX.NS","SUVEN.NS","SUVENPHAR.NS","SUZLON.NS","SWANENERGY.NS","SWARAJENG.NS","SYMPHONY.NS","SYNCOM.NS","SYNGENE.NS","SYRMA.NS","TAJGVK.NS","TALBROAUTO.NS","TMB.NS","TNPL.NS","TNPETRO.NS","TANFACIND.NS","TANLA.NS","TARC.NS","TARSONS.NS","TASTYBITE.NS","TATACHEM.NS","TATAINVEST.NS","TTML.NS","TATVA.NS","TCIEXP.NS","TCNSBRANDS.NS","TCPLPACK.NS","TDPOWERSYS.NS","TEAMLEASE.NS","TECHNOE.NS","TIIL.NS","TEGA.NS","TEJASNET.NS","TEXRAIL.NS","TGVSL.NS","THANGAMAYL.NS","ANUP.NS","THEMISMED.NS","THERMAX.NS","TIRUMALCHM.NS","THOMASCOOK.NS","THYROCARE.NS","TI.NS","TIMETECHNO.NS","TIMEXWATCH.NS","TIMKEN.NS","TIPSINDLTD.NS","TITAGARH.NS","TFCILTD.NS","TRACXN.NS","TRIL.NS","TRANSIND.RE.NS","TRANSPEK.NS","TCI.NS","TRIDENT.NS","TRIVENI.NS","TRITURBINE.NS","TRU.NS","TTKHLTCARE.NS","TTKPRESTIG.NS","TVTODAY.NS","TV18BRDCST.NS","TVSELECT.NS","TVS.NS","TVSSRICHAK.NS","TVSSCS.NS","UDAICEMENT.NS","UFLEX.NS","UGARSUGAR.NS","UGROCAP.NS","UJJIVANSFB.NS","ULTRAMAR.NS","UNICHEMLAB.NS","UNIPARTS.NS","UNIVCABLES.NS","UDSERV.NS","USHAMART.NS","UTIAMC.NS","UTKARSHPB.NS","UTTAMSUGAR.NS","VGUARD.NS","VMART.NS","VSTTILLERS.NS","WABAG.NS","VADILALIND.NS","VAIBHAVGBL.NS","VAKRANGEE.NS","VALIANTORG.NS","DBREALTY.NS","VSSL.NS","VTL.NS","VARROC.NS","VASCONEQ.NS","VENKEYS.NS","VENUSPIPES.NS","VERANDA.NS","VESUVIUS.NS","VIDHIING.NS","VIJAYA.NS","VIKASLIFE.NS","VIMTALABS.NS","VINATIORGA.NS","VINDHYATEL.NS","VINYLINDIA.NS","VIPIND.NS","VISAKAIND.NS","VISHNU.NS","VISHNUPR.NS","VLSFINANCE.NS","VOLTAMP.NS","VRLLOG.NS","VSTIND.NS","WAAREE.NS","WARDINMOBI.NS","WELCORP.NS","WELENT.NS","WELSPUNLIV.NS","WELSPLSOL.NS","WENDT.NS","WSTCSTPAPR.NS","WESTLIFE.NS","WOCKPHARMA.NS","WONDERLA.NS","WPIL.NS","XCHANGING.NS","YASHO.NS","YATHARTH.NS","YATRA.NS","YUKEN.NS","ZAGGLE.NS","ZEEMEDIA.NS","ZENTEC.NS","ZENSARTECH.NS","ZFCVINDIA.NS","ZUARI.NS","ZYDUSWELL.NS"]
 
   
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from scipy.signal import find_peaks, cwt, ricker, hilbert
import nolds
from scipy.stats import zscore

# Sidebar for user input
st.sidebar.subheader("Prediction")

submenu = st.sidebar.selectbox("Select Option", ["Trend", "Price"])  

tickers = st.sidebar.multiselect("Enter Stock Symbols", options=bse_largecap+bse_midcap+bse_smallcap)
time_period = st.sidebar.selectbox("Select Time Period", options=["6mo", "1y", "5y"], index=0)

if submenu == "Trend":
    @st.cache_data
    def load_data(ticker, period):
        return yf.download(ticker, period=period)

    def remove_outliers(df, column='Close', z_thresh=3):
        df['zscore'] = zscore(df[column])
        df = df[df['zscore'].abs() <= z_thresh]
        df.drop(columns='zscore', inplace=True)
        return df

    def calculate_indicators(df):
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_Hist'] = ta.trend.macd_diff(df['Close'])
        df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'])
        df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'])
        df['Stochastic_Oscillator'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['VPT'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
        df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
        df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
        return df

    def calculate_peaks_troughs(df):
        peaks, _ = find_peaks(df['Close'])
        troughs, _ = find_peaks(-df['Close'])
        df['Peaks'] = np.nan
        df['Troughs'] = np.nan
        df.loc[df.index[peaks], 'Peaks'] = df['Close'].iloc[peaks]
        df.loc[df.index[troughs], 'Troughs'] = df['Close'].iloc[troughs]
        return df

    def calculate_fourier(df, n=5):
        close_fft = np.fft.fft(df['Close'].values)
        fft_df = pd.DataFrame({'fft': close_fft})
        fft_df['absolute'] = np.abs(fft_df['fft'])
        fft_df['angle'] = np.angle(fft_df['fft'])
        fft_df = fft_df.sort_values(by='absolute', ascending=False).head(n)
        
        # Calculate dominant frequency in days
        freqs = np.fft.fftfreq(len(df))
        dominant_freqs = freqs[np.argsort(np.abs(close_fft))[-n:]]
        # Ensure frequencies are positive
        positive_freqs = np.abs(dominant_freqs)
        cycles_in_days = 1 / positive_freqs
        fft_df['cycles_in_days'] = cycles_in_days
        return fft_df

    def calculate_hurst(df):
        if len(df) < 100:
            return np.nan
        H = nolds.hurst_rs(df['Close'])
        return H

    def calculate_wavelet(df):
        widths = np.arange(1, 31)
        cwt_matrix = cwt(df['Close'], ricker, widths)
        return cwt_matrix

    def calculate_hilbert(df):
        analytic_signal = hilbert(df['Close'])
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        return amplitude_envelope, instantaneous_phase

    # Streamlit UI
    st.title("Multi-Stock Cycle Detection and Analysis")

    results = []

    for ticker in tickers:
        st.subheader(f"Analysis for {ticker}")
        data = load_data(ticker, time_period)
        
        if not data.empty:
            df = data.copy()
            df = remove_outliers(df)
            df = calculate_indicators(df)
            df = calculate_peaks_troughs(df)

            st.subheader(f"{ticker} Data and Indicators ({time_period} Period)")
            st.dataframe(df.tail())

            # Fourier Transform results
            st.subheader("Fourier Transform - Dominant Cycles")
            fft_df = calculate_fourier(df)
            st.write("Dominant Cycles from Fourier Transform (Top 5):")
            st.dataframe(fft_df)

            # Determine current position in the cycle
            dominant_cycle = fft_df.iloc[1] if fft_df.iloc[1]['absolute'] > fft_df.iloc[2]['absolute'] else fft_df.iloc[2]
            current_phase_angle = dominant_cycle['angle']
            current_position = 'upward' if current_phase_angle > 0 else 'downward'

            st.subheader("Current Cycle Position")
            st.write(f"The stock is currently in an '{current_position}' phase of the dominant cycle with a phase angle of {current_phase_angle:.2f} radians.")

            # Explanation of what this means
            if current_position == 'upward':
                st.write("This means the stock price is currently in an upward trend within its dominant cycle, and it is expected to continue rising.")
            else:
                st.write("This means the stock price is currently in a downward trend within its dominant cycle, and it is expected to continue falling.")

            # Display Hurst Exponent
            st.subheader("Hurst Exponent")
            hurst_value = calculate_hurst(df)
            st.write(f"Hurst Exponent: {hurst_value}")

            # Interpretation of Hurst Exponent
            if hurst_value > 0.5:
                st.write("The time series is trending.")
            elif hurst_value < 0.5:
                st.write("The time series is mean-reverting.")
            else:
                st.write("The time series is a random walk.")

            # Calculate and display Wavelet Transform results
            st.subheader("Wavelet Transform")
            cwt_matrix = calculate_wavelet(df)

            # Insights from Wavelet Transform
            max_wavelet_amplitude = np.max(np.abs(cwt_matrix))
            st.write("**Wavelet Transform Insights:**")
            st.write(f"The maximum amplitude in the wavelet transform is {max_wavelet_amplitude:.2f}.")

            # Calculate and display Hilbert Transform results
            st.subheader("Hilbert Transform")
            amplitude_envelope, instantaneous_phase = calculate_hilbert(df)

            # Insights from Hilbert Transform
            st.write("**Hilbert Transform Insights:**")
            st.write(f"The current amplitude envelope is {amplitude_envelope[-1]:.2f}.")
            st.write(f"The current instantaneous phase is {instantaneous_phase[-1]:.2f} radians.")

            # Collecting results for comparison
            results.append({
                'Ticker': ticker,
                'Fourier_Dominant_Cycle_Days': fft_df['cycles_in_days'].iloc[0],
                'Fourier_Angle': fft_df['angle'].iloc[1],  # Correctly selecting the dominant cycle angle
                'Fourier_Trend': current_position,
                'Hurst_Exponent': hurst_value,
                'Wavelet_Max_Amplitude': max_wavelet_amplitude,
                'Hilbert_Amplitude': amplitude_envelope[-1],
                'Hilbert_Instantaneous_Phase': instantaneous_phase[-1]
            })

    # Comparison and Final Recommendation
    st.subheader("Comparison and Final Recommendation")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Final Recommendation based on collected results
    buy_recommendation = results_df[
        (results_df['Hurst_Exponent'] > 0.5) &  # Trending
        (results_df['Wavelet_Max_Amplitude'] > 1000) &  # Adjust threshold as necessary
        (results_df['Hilbert_Amplitude'] > results_df['Hilbert_Amplitude'].mean()) &  # Strong current price movement
        (results_df['Fourier_Trend'] == 'upward')  # Current price trend is upward
    ]

    if not buy_recommendation.empty:
        st.write("**Recommendation: Buy**")
        st.dataframe(buy_recommendation[['Ticker', 'Fourier_Dominant_Cycle_Days', 'Fourier_Angle', 'Fourier_Trend', 'Hurst_Exponent', 'Wavelet_Max_Amplitude', 'Hilbert_Amplitude']])
    else:
        st.write("No strong buy recommendations based on the current analysis.")

else:
    pass
