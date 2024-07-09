import os
import streamlit as st
import random
import string
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound


import bcrypt  # for password hashing
from dotenv import load_dotenv
from password_validator import PasswordValidator

import yfinance as yf
import pandas as pd
import pandas_ta as pta
import numpy as np
import plotly.graph_objs as go
import ta
from scipy.stats import linregress
from functools import lru_cache
import smtplib
import mplfinance as mpf
import plotly.express as px
from scipy.signal import find_peaks, cwt, ricker, hilbert

from scipy.stats import zscore


 # List of stock tickers
bse_largecap = ["ABB.BO","ADANIENG.BO","ADANIENT.BO","ADANIGREEN.BO","ADANIPORTS.BO","ADANIPOWER.BO","ATGL.NS","AWL.NS","AMBUJACEM.BO","APOLLOHOSP.BO","ASIANPAINT.BO","DMART.BO","AXISBANK.BO","BAJAJ-AUTO.BO","BAJFINANCE.BO","BAJAJFINSV.BO","BAJAJHLDNG.BO","BANDHANBNK.BO","BANKBARODA.BO","BERGEPAINT.BO","BEL.BO","BPCL.BO","BHARTIARTL.BO","BOSCHLTD.BO","BRITANNIA.BO","CHOLAFIN.BO","CIPLA.BO","COALINDIA.BO","DABUR.BO","DIVISLAB.BO","DLF.BO","DRREDDY.BO","EICHERMOT.BO","NYKAA.NS","GAIL.BO","GODREJCP.BO","GRASIM.BO","HAVELLS.BO","HCLTECH.BO","HDFCAMC.BO","HDFCBANK.BO","HDFCLIFE.BO","HEROMOTOCO.BO","HINDALCO.BO","HAL.BO","HINDUNILVR.BO","HINDZINC.BO","ICICIBANK.BO","ICICIGI.BO","ICICIPRULI.BO","IOC.BO","INDUSTOWER.BO","INDUSINDBK.BO","NAUKRI.BO","INFY.BO","INDIGO.BO","ITC.BO","JIOFIN.NS","JSWSTEEL.BO","KOTAKBANK.BO","LT.BO","LICI.NS","MINDTREE.BO","M&M.BO","MANKIND.NS","MARICO.BO","MARUTI.BO","NESTLEIND.BO","NTPC.BO","ONGC.BO","PAYTM.NS","PIDILITIND.BO","POWERGRID.BO","PNB.BO","RELIANCE.BO","SBICARD.BO","SBILIFE.BO","SHREECEM.BO","SIEMENS.BO","SRF.BO","SBIN.BO","SUNPHARMA.BO","TCS.BO","TATACONSUM.BO","TATAMOTORS.BO","TATAMTRDVR.BO","TATAPOWER.BO","TATASTEEL.BO","TECHM.BO","TITAN.BO","ULTRACEMCO.BO","MCDOWELL-N.BO","UPL.BO","VBL.BO","VEDL.BO","WIPRO.BO","ZOMATO.BO","ZYDUSLIFE.NS"]
bse_midcap = ["3MINDIA.BO","AARTIIND.BO","ABBOTINDIA.BO","ACC.BO","ABCAPITAL.BO","ABFRL.BO","AJANTPHARM.BO","ALKEM.BO","APLAPOLLO.BO","ASHOKLEY.BO","ASTRAL.BO","AUBANK.BO","AUROPHARMA.BO","BALKRISIND.BO","BANKINDIA.BO","BAYERCROP.BO","BHARATFORG.BO","BHEL.BO","BIOCON.BO","CANBK.BO","CASTROLIND.BO","CGPOWER.BO","CLEANSCIENCE.BO","COLPAL.BO","CONCOR.BO","COROMANDEL.BO","CRISIL.BO","CROMPTON.BO","CUMMINSIND.BO","DALBHARAT.BO","DEEPAKNTR.BO","DELHIVERY.NS","EMAMILTD.BO","ENDURANCE.BO","EXIDEIND.BO","FEDERALBNK.BO","GICRE.BO","GILLETTE.BO","GLAND.BO","GLAXO.BO","GLENMARK.BO","GMRINFRA.BO","GODREJIND.BO","GODREJPROP.BO","GUJFLUORO.BO","GUJGASLTD.BO","HINDPETRO.BO","HONAUT.BO","ISEC.BO","IDBI.BO","IDFCFIRSTB.BO","INDIANB.BO","INDHOTEL.BO","IOB.BO","IRCTC.BO","IRFC.BO","IREDAL.BO","IGL.BO","IPCALAB.BO","JINDALSTEL.BO","JSWENERGY.BO","JSWINFRA.NS","JUBLFOOD.BO","KANSAINER.BO","L&TFH.BO","LTTS.BO","LAURUSLABS.BO","LICHSGFIN.BO","LINDEINDIA.BO","LUPIN.BO","LODHA.BO","M&MFIN.BO","MFSL.BO","MAXHEALTH.BO","MPHASIS.BO","MRF.BO","MUTHOOTFIN.BO","NHPC.BO","NAM-INDIA.BO","NMDC.BO","NUVOCO.BO","OBEROIRLTY.BO","OIL.BO","OFSS.BO","PAGEIND.BO","PATANJALI.NS","PAYTM.BO","PERSISTENT.BO","PETRONET.BO","PIIND.BO","PEL.BO","POLYCAB.BO","PFC.BO","PGHH.BO","RAJESHEXPO.BO","RAMCOCEM.BO","RECLTD.BO","RELAXO.BO","SAMARTI.BO","SCHAEFFLER.BO","SRTRANSFIN.BO","SJVN.BO","SOLARINDS.BO","SONACOMS.BO","STARHEALTH.NS","SAIL.BO","SUNTV.BO","SUPREMEIND.BO","TATACOMM.BO","TATAELXSI.BO","TATATECH.BO","NIACL.BO","TORNTPHARM.BO","TORNTPOWER.BO","TRENT.BO","TIINDIA.BO","TVSMOTOR.BO","UCOBANK.BO","UNIONBANK.BO","UBL.BO","UNOMINDA.BO","VEDL.BO","IDEA.BO","VOLTAS.BO","WHIRLPOOL.BO","YESBANK.BO","ZEEL.BO"]
bse_smallcap = ["360ONE.NS","3IINFOTECH.NS","5PAISA.NS","63MOONS.NS","AARTIDRUGS.NS","AARTIPHARM.NS","AAVAS.NS","AHL.NS","ACCELYA.NS","ACE.NS","ADFFOODS.NS","ABSLAMC.NS","AVL.BO","ADORWELD.NS","ADVENZYMES.NS","AEGISCHEM.NS","AEROFLEX.NS","AETHER.NS","AFFLE.NS","AGARIND.NS","AGIGREEN.NS","ATFL.NS","AGSTRA.NS","AHLUCONT.NS","AIAENG.NS","AJMERA.NS","AKZOINDIA.NS","ALEMBICLTD.NS","APLLTD.NS","ALICON.NS","ALKYLAMINE.NS","ACLGATI.NS","ALLCARGO.NS","ATL.NS","ALLSEC.NS","ALOKINDS.NS","AMARAJABAT.NS","AMBER.NS","AMBIKCO.NS","AMIORG.NS","ANANDRATHI.NS","ANANTRAJ.NS","ANDHRAPAP.NS","ANDHRAPET.NS","ANDREWYU.NS","ANGELONE.NS","AWHCL.NS","ANURAS.NS","APARINDS.NS","APCOTEXIND.NS","APOLLO.NS","APOLLOPIPE.NS","APOLLOTYRE.NS","APTECHT.NS","APTUS.NS","ARCHIECHEM.NS","ARIHANTCAP.NS","ARIHANTSUP.NS","ARMANFIN.NS","ARTEMISMED.NS","ARVINDFASN.NS","ARVIND.NS","ARVSMART.NS","ASAHIINDIA.NS","ASHAPURMIN.NS","ASHIANA.NS","ASHOKA.NS","ASIANTILES.NS","ASKAUTOLTD.NS","ASALCBR.NS","ASTEC.NS","ASTERDM.NS","ASTRAMICRO.NS","ASTRAZEN.NS","ATULAUTO.NS","ATUL.NS","AURIONPRO.NS","AIIL.BO","AUTOAXLES.NS","ASAL.NS","AVADHSUGAR.NS","AVALON.NS","AVANTEL.NS","AVANTIFEED.NS","AVTNPL.NS","AXISCADES.NS","AZAD.NS","BLKASHYAP.NS","BAJAJCON.NS","BAJAJELEC.NS","BAJAJHCARE.NS","BAJAJHIND.NS","BALAMINES.NS","BALMLAWRIE.NS","BLIL.BO","BALRAMCHIN.NS","BALUFORGE.BO","BANCOINDIA.NS","MAHABANK.NS","BANARISUG.NS","BARBEQUE.NS","BASF.NS","BATAINDIA.NS","BCLIND.NS","BLAL.NS","BEML.NS","BESTAGRO.NS","BFINVEST.NS","BFUTILITIE.NS","BHAGERIA.NS","BHAGCHEM.NS","BEPL.NS","BHARATBIJ.NS","BDL.NS","BHARATWIRE.NS","BIGBLOC.NS","BIKAJI.NS","BIRLACORPN.NS","BSOFT.NS","BLACKBOX.NS","BLACKROSE.NS","BLISSGVS.NS","BLS.NS","BLUEDART.NS","BLUEJET.NS","BLUESTARCO.NS","BODALCHEM.NS","BOMDYEING.NS","BOROLTD.NS","BORORENEW.NS","BRIGADE.NS","BUTTERFLY.NS","CEINFO.NS","CAMLINFINE.NS","CAMPUS.NS","CANFINHOME.NS","CANTABIL.NS","CAPACITE.NS","CAPLIPOINT.NS","CGCL.NS","CARBORUNIV.NS","CARERATING.NS","CARTRADE.NS","CARYSIL.NS","CCL.NS","CEATLTD.NS","CELLO.NS","CENTRALBK.NS","CENTRUM.NS","CENTUM.NS","CENTENKA.NS","CENTURYPLY.NS","CENTURYTEX.NS","CERA.NS","CESC.NS","CHALET.NS","CLSEL.NS","CHAMBLFERT.NS","CHEMCON.NS","CHEMPLASTS.NS","CHENNPETRO.NS","CHEVIOT.NS","CHOICEIN.NS","CHOLAHOLD.NS","CIEINDIA.NS","CIGNITITEC.NS","CUB.NS","CMSINFO.NS","COCHINSHIP.NS","COFFEEDAY.NS","COFORGE.NS","CAMS.NS","CONCORDBIO.NS","CONFIPET.NS","CONTROLPR.NS","COSMOFIRST.NS","CRAFTSMAN.NS","CREDITACC.NS","MUFTI.BO","CRESSANDA.NS","CSBBANK.NS","CYIENTDLM.NS","CYIENT.NS","DLINKINDIA.NS","DALMIASUG.NS","DATAPATTNS.NS","DATAMATICS.NS","DBCORP.NS","DCBBANK.NS","DCMSHRIRAM.NS","DCW.NS","DCXINDIA.NS","DDEVPLASTIK.BO","DECCANCE.NS","DEEPINDS.NS","DEEPAKFERT.NS","DELTACORP.NS","DEN.NS","DEVYANI.NS","DHAMPURBIO.NS","DHAMPURSUG.NS","DHANI.NS","DHANUKA.NS","DHARAMSI.NS","DCGL.NS","DHUNINV.NS","DBL.NS","DISHTV.NS","DISHMAN.NS","DIVGI.NS","DIXON.NS","DODLA.NS","DOLLAR.NS","DOMS.NS","LALPATHLAB.NS","DREAMFOLKS.NS","DREDGECORP.NS","DWARKESH.NS","DYNAMATECH.NS","EASEMYTRIP.NS","ECLERX.NS","EDELWEISS.NS","EIDPARRY.NS","EIHAHOTELS.NS","EIHOTEL.NS","EKIEGS.NS","ELECON.NS","EMIL.NS","ELECTCAST.NS","ELGIEQUIP.NS","ELIN.NS","ELPROINTL.NS","EMAMIPAP.NS","EMSLIMITED.NS","EMUDHRA.NS","ENGINERSIN.NS","ENIL.NS","EPIGRAL.NS","EPL.NS","EQUITASBNK.NS","ERIS.NS","ESABINDIA.NS","ESAFSFB.NS","ESCORTS.NS","ESTER.NS","ETHOSLTD.NS","EUREKAFORBE.BO","EVEREADY.NS","EVERESTIND.NS","EKC.NS","EXCELINDUS.NS","EXPLEOSOL.NS","FAIRCHEM.NS","FAZETHREE.NS","FDC.NS","FEDFINA.NS","FMGOETZE.NS","FIEMIND.NS","FILATEX.NS","FINEORG.NS","FCL.NS","FINOPB.NS","FINCABLES.NS","FINPIPE.NS","FSL.NS","FIVESTAR.NS","FLAIR.NS","FOODSIN.NS","FORCEMOT.NS","FORTIS.NS","FOSECOIND.NS","FUSION.NS","GMBREW.NS","GNA.NS","GRINFRA.NS","GABRIEL.NS","GALAXYSURF.NS","GALLANTT.NS","GANDHAR.NS","GANDHITUBE.NS","GANESHBENZ.NS","GANESHHOUC.NS","GANECOS.NS","GRSE.NS","GARFILA.NS","GARFIBRES.NS","GDL.NS","GEPIL.NS","GET&D.NS","GENESYS.NS","GENSOL.NS","GENUSPOWER.NS","GEOJITFSL.NS","GFLLIMITED.NS","GHCL.NS","GHCLTEXTIL.NS","GICHSGFIN.NS","GLENMARK.NS","MEDANTA.NS","GSURF.NS","GLOBUSSPR.NS","GMMPFAUDLR.NS","GMRINFRA.NS","GOFASHION.NS","GOCLCORP.NS","GPIL.NS","GODFRYPHLP.NS","GODREJAGRO.NS","GOKEX.NS","GOKUL.NS","GOLDIAM.NS","GOODLUCK.NS","GOODYEAR.NS","GRANULES.NS","GRAPHITE.NS","GRAUWEIL.NS","GRAVITA.NS","GESHIP.NS","GREAVESCOT.NS","GREENLAM.NS","GREENPANEL.NS","GREENPLY.NS","GRINDWELL.NS","GRMOVER.NS","GTLINFRA.NS","GTPL.NS","GUFICBIO.NS","GUJALKALI.NS","GAEL.NS","GIPCL.NS","GMDCLTD.NS","GNFC.NS","GPPL.NS","GSFC.NS","GSPL.NS","GUJTHEMIS.NS","GULFOILLUB.NS","GULPOLY.NS","HGINFRA.NS","HAPPSTMNDS.NS","HAPPYFORGE.NS","HARDWYN.NS","HARIOMPIPE.NS","HARSHAENG.NS","HATHWAY.NS","HATSUN.NS","HAWKINCOOK.BO","HBLPOWER.NS","HCG.NS","HEG.NS","HEIDELBERG.NS","HEMIPROP.NS","HERANBA.NS","HERCULES.NS","HERITGFOOD.NS","HESTERBIO.NS","HEUBACHIND.NS","HFCL.NS","HITECH.NS","HIKAL.NS","HIL.NS","HSCL.NS","HIMATSEIDE.NS","HGS.NS","HCC.NS","HINDCOPPER.NS","HINDUSTAN.NS","HINDOILEXP.NS","HINDWARE.NS","POWERINDIA.NS","HLEGLAS.NS","HOTELLEELA.NS","HMAAGRO.NS","HOMEFIRST.NS","HONASA.NS","HONDAPOWER.NS","HUDCO.NS","HPAL.NS","HPL.NS","HUHTAMAKI.NS","ICRA.NS","IDEA.NS","IDFC.NS","IFBIND.NS","IFCI.NS","IFGLEXPOR.NS","IGPL.NS","IGARASHI.NS","IIFL.NS","IIFLSEC.NS","IKIO.NS","IMAGICAA.NS","INDIACEM.NS","INDIAGLYCO.NS","INDNIPPON.NS","IPL.NS","INDIASHLTR.NS","ITDC.NS","IBULHSGFIN.NS","IBREALEST.NS","INDIAMART.NS","IEX.NS","INDIANHUME.NS","IMFA.NS","INDIGOPNTS.NS","INDOAMIN.NS","ICIL.NS","INDORAMA.NS","INDOCO.NS","INDREMEDI.NS","INFIBEAM.NS","INFOBEAN.NS","INGERRAND.NS","INNOVACAP.NS","INOXGREEN.NS","INOXINDIA.NS","INOXWIND.NS","INSECTICID.NS","INTELLECT.NS","IOLCP.NS","IONEXCHANG.NS","IRB.NS","IRCON.NS","IRMENERGY.NS","ISGEC.NS","ITDCEM.NS","ITI.NS","JKUMAR.NS","JKTYRE.NS","JBCHEPHARM.NS","JKCEMENT.NS","JAGRAN.NS","JAGSNPHARM.NS","JAIBALAJI.NS","JAICORPLTD.NS","JISLJALEQS.NS","JPASSOCIAT.NS","JPPOWER.NS","J&KBANK.NS","JAMNAAUTO.NS","JAYBARMARU.NS","JAYAGROGN.NS","JAYNECOIND.NS","JBMA.NS","JINDRILL.NS","JINDALPOLY.NS","JINDALSAW.NS","JSL.NS","JINDWORLD.NS","JKLAKSHMI.NS","JKPAPER.NS","JMFINANCIL.NS","JCHAC.NS","JSWHL.NS","JTEKTINDIA.NS","JTLIND.NS","JUBLINDS.NS","JUBLINGREA.NS","JUBLPHARMA.NS","JLHL.NS","JWL.NS","JUSTDIAL.NS","JYOTHYLAB.NS","JYOTIRES.BO","KABRAEXTRU.NS","KAJARIACER.NS","KALPATPOWR.NS","KALYANKJIL.NS","KALYANIFRG.NS","KSL.NS","KAMAHOLD.NS","KAMDHENU.NS","KAMOPAINTS.NS","KANORICHEM.NS","KTKBANK.NS","KSCL.NS","KAYNES.NS","KDDL.NS","KEC.NS","KEI.NS","KELLTONTEC.NS","KENNAMET.NS","KESORAMIND.NS","KKCL.NS","RUSTOMJEE.NS","KFINTECH.NS","KHAICHEM.NS","KINGFA.NS","KIOCL.NS","KIRIINDUS.NS","KIRLOSBROS.NS","KIRLFER.NS","KIRLOSIND.NS","KIRLOSENG.NS","KIRLPNU.NS","KITEX.NS","KMCSHIL.BO","KNRCON.NS","KOKUYOCMLN.NS","KOLTEPATIL.NS","KOPRAN.NS","KOVAI.NS","KPIGREEN.NS","KPITTECH.NS","KPRMILL.NS","KRBL.NS","KIMS.NS","KRSNAA.NS","KSB.NS","KSOLVES.NS","KUANTUM.NS","LAOPALA.NS","LAXMIMACH.NS","LANCER.NS","LANDMARK.NS","LATENTVIEW.NS","LXCHEM.NS","LEMONTREE.NS","LGBBROSLTD.NS","LIKHITHA.NS","LINCPEN.NS","LINCOLN.NS","LLOYDSENGG.NS","LLOYDSENT.BO","LLOYDSME.NS","DAAWAT.NS","LUMAXTECH.NS","LUMAXIND.NS","MMFL.NS","MKPL.NS","MCLOUD.BO","MGL.NS","MTNL.NS","MAHSCOOTER.NS","MAHSEAMLES.NS","MHRIL.NS","MAHLIFE.NS","MAHLOG.NS","MANINDS.NS","MANINFRA.NS","MANGCHEFER.NS","MANAKSIA.NS","MANALIPETC.NS","MANAPPURAM.NS","MANGLMCEM.NS","MRPL.NS","MVGJL.NS","MANORAMA.NS","MARATHON.NS","MARKSANS.NS","MASFIN.NS","MASTEK.NS","MATRIMONY.NS","MAXESTATE.NS","MAYURUNIQ.NS","MAZDOCK.NS","MEDICAMEQ.NS","MEDPLUS.NS","MEGH.NS","MENONBE.NS","METROBRAND.NS","METROPOLIS.NS","MINDACORP.NS","MIRZAINT.NS","MIDHANI.NS","MISHTANN.NS","MMTC.NS","MOIL.NS","MOLDTKPAC.NS","MONARCH.NS","MONTECARLO.NS","MOREPENLAB.NS","MOSCHIP.NS","MSUMI.NS","MOTILALOFS.NS","MPSLTD.NS","BECTORFOOD.NS","MSTCLTD.NS","MTARTECH.NS","MUKANDLTD.NS","MCX.NS","MUTHOOTMF.NS","NACLIND.NS","NAGAFERT.NS","NAHARINV.NS","NAHARSPING.NS","NSIL.NS","NH.NS","NATCOPHARM.NS","NATIONALUM.NS","NFL.NS","NAVINFLUOR.NS","NAVKARCORP.NS","NAVNETEDUL.NS","NAZARA.NS","NBCC.NS","NCC.NS","NCLIND.NS","NELCAST.NS","NELCO.NS","NEOGEN.NS","NESCO.NS","NETWEB.NS","NETWORK18.NS","NEULANDLAB.NS","NDTV.NS","NEWGEN.NS","NGLFINE.NS","NIITMTS.NS","NIITLTD.NS","NIKHILADH.NS","NILKAMAL.NS","NITINSPIN.NS","NITTAGELA.BO","NLCINDIA.NS","NSLNISP.NS","NOCIL.NS","NOVARTIND.NS","NRBBEARING.NS","NUCLEUS.NS","NUVAMA.NS","OLECTRA.NS","OMAXE.NS","ONMOBILE.NS","ONWARDTEC.NS","OPTIEMUS.NS","ORIENTBELL.NS","ORIENTCEM.NS","ORIENTELEC.NS","GREENPOWER.NS","ORIENTPPR.NS","OAL.NS","OCCL.NS","ORIENTHOT.NS","OSWALGREEN.NS","PAISALO.NS","PANACEABIO.NS","PANAMAPET.NS","PARADEEP.NS","PARAGMILK.NS","PARA.NS","PARAS.NS","PATELENG.NS","PAUSHAKLTD.NS","PCJEWELLER.NS","PCBL.NS","PDSL.NS","PGIL.NS","PENIND.NS","PERMAGNET.NS","PFIZER.NS","PGEL.NS","PHOENIXLTD.NS","PILANIINVS.NS","PPLPHARMA.NS","PITTILAM.NS","PIXTRANS.NS","PNBGILTS.NS","PNBHOUSING.NS","PNCINFRA.NS","POKARNA.NS","POLYMED.NS","POLYPLEX.NS","POONAWALLA.NS","POWERMECH.NS","PRAJIND.NS","PRAKASHSTL.NS","DIAMONDYD.NS","PRAVEG.NS","PRECAM.NS","PRECWIRE.NS","PRESTIGE.NS","PRICOLLTD.NS","PFOCUS.NS","PRIMO.NS","PRINCEPIPE.NS","PRSMJOHNSN.NS","PRIVISCL.NS","PGHL.NS","PROTEAN.BO","PRUDENT.NS","PSPPROJECT.NS","PFS.NS","PTC.NS","PTCIL.BO","PSB.NS","PUNJABCHEM.NS","PURVA.NS","PVRINOX.NS","QUESS.NS","QUICKHEAL.NS","QUINT.NS","RRKABEL.NS","RSYSTEMS.NS","RACLGEAR.NS","RADIANT.NS","RADICO.NS","RAGHAVPRO.NS","RVNL.NS","RAILTEL.NS","RAIN.NS","RAINBOW.NS","RAJRATAN.NS","RALLIS.NS","RRWL.NS","RAMASTEEL.NS","RAMCOIND.NS","RAMCOSYS.NS","RKFORGE.NS","RAMKY.NS","RANEHOLDIN.NS","RML.NS","RCF.NS","RATEGAIN.NS","RATNAMANI.NS","RTNINDIA.NS","RTNPOWER.NS","RAYMOND.NS","RBLBANK.NS","REDINGTON.NS","REDTAPE.NS","REFEX.NS","RIIL.NS","RELINFRA.NS","RPOWER.NS","RELIGARE.NS","RENAISSANCE.NS","REPCOHOME.NS","REPRO.NS","RESPONIND.NS","RBA.NS","RHIM.NS","RICOAUTO.NS","RISHABHINS.NS","RITES.NS","ROLEXRINGS.NS","ROSSARI.NS","ROSSELLIND.NS","ROTOPUMPS.NS","ROUTE.NS","ROHL.NS","RPGLIFE.NS","RPSGVENT.NS","RSWM.NS","RUBYMILLS.NS","RUPA.NS","RUSHIL.NS","SCHAND.NS","SHK.NS","SJS.NS","SPAL.NS","SADHANANIQ.NS","SAFARI.NS","SAGCEM.NS","SAILKALAM.NS","SALASAR.NS","SAMHIHOTEL.NS","SANDHAR.NS","SANDUMA.NS","SANGAMIND.NS","SANGHIIND.NS","SANGHVIMOV.NS","SANMITINFRA.NS","SANOFI.NS","SANSERA.NS","SAPPHIRE.NS","SARDAEN.NS","SAREGAMA.NS","SASKEN.NS","SASTASUNDR.NS","SATINDLTD.NS","SATIA.NS","SATIN.NS","SOTL.NS","SBFC.NS","SCHNEIDER.NS","SEAMECLTD.NS","SENCO.NS","SEPC.NS","SEQUENT.NS","SESHAPAPER.NS","SGFIN.NS","SHAILY.NS","SHAKTIPUMP.NS","SHALBY.NS","SHALPAINTS.NS","SHANKARA.NS","SHANTIGEAR.NS","SHARDACROP.NS","SHARDAMOTR.NS","SHAREINDIA.NS","SFL.NS","SHILPAMED.NS","SCI.NS","SHIVACEM.NS","SBCL.NS","SHIVALIK.NS","SHOPERSTOP.NS","SHREDIGCEM.NS","SHREEPUSHK.NS","RENUKA.NS","SHREYAS.NS","SHRIRAMPPS.NS","SHYAMMETL.NS","SIGACHI.NS","SIGNA.NS","SIRCA.NS","SIS.NS","SIYSIL.NS","SKFINDIA.NS","SKIPPER.NS","SMCGLOBAL.NS","SMLISUZU.NS","SMSPHARMA.NS","SNOWMAN.NS","SOBHA.NS","SOLARA.NS","SDBL.NS","SOMANYCERA.NS","SONATSOFTW.NS","SOUTHBANK.NS","SPANDANA.NS","SPECIALITY.NS","SPENCERS.NS","SPICEJET.NS","SPORTKING.NS","SRHHYPOLTD.NS","STGOBANQ.NS","STARCEMENT.NS","STEELXIND.NS","SSWL.NS","STEELCAS.NS","SWSOLAR.NS","STERTOOLS.NS","STRTECH.NS","STOVEKRAFT.NS","STAR.NS","STYLAMIND.NS","STYRENIX.NS","SUBEXLTD.NS","SUBROS.NS","SUDARSCHEM.NS","SUKHJITS.NS","SULA.NS","SUMICHEM.NS","SUMMITSEC.NS","SPARC.NS","SUNCLAYLTD.NS","SUNDRMFAST.NS","SUNFLAG.NS","SUNTECK.NS","SUPRAJIT.NS","SUPPETRO.NS","SUPRIYA.NS","SURAJEST.NS","SURYAROSNI.NS","SURYODAY.NS","SUTLEJTEX.NS","SUVEN.NS","SUVENPHAR.NS","SUZLON.NS","SWANENERGY.NS","SWARAJENG.NS","SYMPHONY.NS","SYNCOM.NS","SYNGENE.NS","SYRMA.NS","TAJGVK.NS","TALBROAUTO.NS","TMB.NS","TNPL.NS","TNPETRO.NS","TANFACIND.NS","TANLA.NS","TARC.NS","TARSONS.NS","TASTYBITE.NS","TATACHEM.NS","TATAINVEST.NS","TTML.NS","TATVA.NS","TCIEXP.NS","TCNSBRANDS.NS","TCPLPACK.NS","TDPOWERSYS.NS","TEAMLEASE.NS","TECHNOE.NS","TIIL.NS","TEGA.NS","TEJASNET.NS","TEXRAIL.NS","TGVSL.NS","THANGAMAYL.NS","ANUP.NS","THEMISMED.NS","THERMAX.NS","TIRUMALCHM.NS","THOMASCOOK.NS","THYROCARE.NS","TI.NS","TIMETECHNO.NS","TIMEXWATCH.NS","TIMKEN.NS","TIPSINDLTD.NS","TITAGARH.NS","TFCILTD.NS","TRACXN.NS","TRIL.NS","TRANSIND.RE.NS","TRANSPEK.NS","TCI.NS","TRIDENT.NS","TRIVENI.NS","TRITURBINE.NS","TRU.NS","TTKHLTCARE.NS","TTKPRESTIG.NS","TVTODAY.NS","TV18BRDCST.NS","TVSELECT.NS","TVS.NS","TVSSRICHAK.NS","TVSSCS.NS","UDAICEMENT.NS","UFLEX.NS","UGARSUGAR.NS","UGROCAP.NS","UJJIVANSFB.NS","ULTRAMAR.NS","UNICHEMLAB.NS","UNIPARTS.NS","UNIVCABLES.NS","UDSERV.NS","USHAMART.NS","UTIAMC.NS","UTKARSHPB.NS","UTTAMSUGAR.NS","VGUARD.NS","VMART.NS","VSTTILLERS.NS","WABAG.NS","VADILALIND.NS","VAIBHAVGBL.NS","VAKRANGEE.NS","VALIANTORG.NS","DBREALTY.NS","VSSL.NS","VTL.NS","VARROC.NS","VASCONEQ.NS","VENKEYS.NS","VENUSPIPES.NS","VERANDA.NS","VESUVIUS.NS","VIDHIING.NS","VIJAYA.NS","VIKASLIFE.NS","VIMTALABS.NS","VINATIORGA.NS","VINDHYATEL.NS","VINYLINDIA.NS","VIPIND.NS","VISAKAIND.NS","VISHNU.NS","VISHNUPR.NS","VLSFINANCE.NS","VOLTAMP.NS","VRLLOG.NS","VSTIND.NS","WAAREE.NS","WARDINMOBI.NS","WELCORP.NS","WELENT.NS","WELSPUNLIV.NS","WELSPLSOL.NS","WENDT.NS","WSTCSTPAPR.NS","WESTLIFE.NS","WOCKPHARMA.NS","WONDERLA.NS","WPIL.NS","XCHANGING.NS","YASHO.NS","YATHARTH.NS","YATRA.NS","YUKEN.NS","ZAGGLE.NS","ZEEMEDIA.NS","ZENTEC.NS","ZENSARTECH.NS","ZFCVINDIA.NS","ZUARI.NS","ZYDUSWELL.NS"]
          
largecap_tickers = [
                "ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO",
                "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS",
                "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS",
                "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS",
                "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS",
                "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS",
                "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS",
                "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS",
                "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS",
                "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS",
                "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO",
                "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ABBOTINDIA.NS", "ADANIPOWER.NS",
                "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO",
                "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO",
                "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS",
                "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO"
            ]

midcap_tickers = [
                "PNCINFRA.NS", "INDIASHLTR.NS", "RAYMOND.NS", "KAMAHOLD.BO", "BENGALASM.BO", "CHOICEIN.NS",
                "GRAVITA.NS", "HGINFRA.NS", "JKPAPER.NS", "MTARTECH.NS", "HAPPSTMNDS.NS", "SARDAEN.NS",
                "WELENT.NS", "LTFOODS.NS", "GESHIP.NS", "SHRIPISTON.NS", "SHAREINDIA.NS", "CYIENTDLM.NS", "VTL.NS",
                "EASEMYTRIP.NS", "LLOYDSME.NS", "ROUTE.NS", "VAIBHAVGBL.NS", "GOKEX.NS", "USHAMART.NS", "EIDPARRY.NS",
                "KIRLOSBROS.NS", "MANINFRA.NS", "CMSINFO.NS", "RALLIS.NS", "GHCL.NS", "NEULANDLAB.NS", "SPLPETRO.NS",
                "MARKSANS.NS", "NAVINFLUOR.NS", "ELECON.NS", "TANLA.NS", "KFINTECH.NS", "TIPSINDLTD.NS", "ACI.NS",
                "SURYAROSNI.NS", "GPIL.NS", "GMDCLTD.NS", "MAHSEAMLES.NS", "TDPOWERSYS.NS", "TECHNOE.NS", "JLHL.NS"
            ]

ftse100_tickers = [
            "III.L", "ADM.L", "AAF.L", "AAL.L", "ANTO.L", "AHT.L", "ABF.L", "AZN.L",
            "AUTO.L", "AV.L", "BME.L", "BA.L", "BARC.L", "BDEV.L", "BEZ.L", "BKGH.L", 
            "BP.L", "BATS.L", "BT.A.L", "BNZL.L", "BRBY.L", "CNA.L", "CCH.L", "CPG.L", 
            "CTEC.L", "CRDA.L", "DARK.L", "DCC.L", "DGE.L", "DPLM.L", "EZJ.L", "ENT.L", 
            "EXPN.L", "FCIT.L", "FRAS.L", "FRES.L", "GLEN.L", "GSK.L", "HLN.L", "HLMA.L", 
            "HL.L", "HIK.L", "HWDN.L", "HSBA.L", "IMI.L", "IMB.L", "INF.L", "IHG.L", 
            "ICP.L", "IAG.L", "ITRK.L", "JD.L", "KGF.L", "LAND.L", "LGEN.L", "LLOY.L", 
            "LSEG.L", "LMP.L", "MNG.L", "MKS.L", "MRO.L", "MNDI.L", "NG.L", "NWG.L", 
            "NXT.L", "PSON.L", "PSH.L", "PSN.L", "PHNX.L", "PRU.L", "RKT.L", "REL.L", 
            "RTO.L", "RMV.L", "RIO.L", "RR.L", "SGE.L", "SBRY.L", "SDR.L", "SMT.L", 
            "SGRO.L", "SVT.L", "SHEL.L", "SN.L", "SMDS.L", "SMIN.L", "SKG.L", "SPX.L", 
            "SSE.L", "STAN.L", "TW.L", "TSCO.L", "ULVR.L", "UTG.L", "UU.L", "VTY.L", 
            "VOD.L", "WEIR.L", "WTB.L", "WPP.L"
            ]

sp500_tickers = [
            "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "ABNB",
            "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN",
            "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH",
            "ADI", "ANSS", "AON", "APA", "AAPL", "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", 
            "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", 
            "BK", "BBWI", "BAX", "BDX", "BRK.B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BX", "BA", 
            "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "BLDR", "BG", "CDNS", 
            "CZR", "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE", 
            "CDW", "CE", "COR", "CNC", "CNP", "CF", "CHRW", "CRL", "SCHW", "CHTR", "CVX", "CMG", 
            "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO", 
            "CTSH", "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY", 
            "CTVA", "CSGP", "COST", "CTRA", "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", 
            "DVA", "DAY", "DECK", "DE", "DAL", "DVN", "DXCM", "FANG", "DLR", "DFS", "DG", "DLTR", 
            "D", "DPZ", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", 
            "EIX", "EW", "EA", "ELV", "LLY", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX", 
            "EQIX", "EQR", "ESS", "EL", "ETSY", "EG", "EVRG", "ES", "EXC", "EXPE", "EXPD", "EXR", 
            "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB", "FSLR", "FE", 
            "FI", "FMC", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE", 
            "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GDDY", 
            "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", 
            "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", 
            "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP", "IPG", "INTU", "ISRG", 
            "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "JNPR", 
            "K", "KVUE", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", 
            "LHX", "LH", "LRCX", "LW", "LVS", "LDOS", "LEN", "LIN", "LYV", "LKQ", "LMT", "L", 
            "LOW", "LULU", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", 
            "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU", 
            "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", 
            "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE", 
            "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", 
            "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PANW", 
            "PARA", "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW", 
            "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA", 
            "PHM", "QRVO", "PWR", "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF", 
            "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC", 
            "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SNA", "SOLV", "SO", "LUV", 
            "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", 
            "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TFX", "TER", "TSLA", "TXN", 
            "TXT", "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", 
            "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VLTO", 
            "VRSN", "VRSK", "VZ", "VRTX", "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW", 
            "WAB", "WBA", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", 
            "WRK", "WY", "WMB", "WTW", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS"
             ]


# Set wide mode as default layout
st.set_page_config(layout="wide", page_title="TradeSense")

# Load environment variables from .env file
load_dotenv()

# Database setup
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    dob = Column(Date, nullable=False)  # Date of birth
    pob = Column(String, nullable=False)  # Place of birth

class Watchlist(Base):
    __tablename__ = 'watchlists'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    ticker = Column(String, nullable=False)
    date_added = Column(Date, default=datetime.utcnow)

class Portfolio(Base):
    __tablename__ = 'portfolios'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    ticker = Column(String, nullable=False)
    shares = Column(Float, nullable=False)
    bought_price = Column(Float, nullable=False)
    date_added = Column(Date, default=datetime.utcnow)

# Create the database session
DATABASE_URL = "sqlite:///new_etrade.db"  # Change the database name here
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)  # This will create the new database

Session = sessionmaker(bind=engine)
session = Session()

# Initialize session state for login status and reset code
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'email' not in st.session_state:
    st.session_state.email = ""
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Password validation schema
password_schema = PasswordValidator()
password_schema \
    .min(8) \
    .max(100) \
    .has().uppercase() \
    .has().lowercase() \
    .has().digits() \
    .has().no().spaces()

# Function to hash password using bcrypt
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

# Function to verify password using bcrypt
def verify_password(hashed_password, plain_password):
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError as e:
        print(f"Error verifying password: {e}")
        return False

# Function to send email (if needed)
def send_email(to_email, subject, body):
    from_email = os.getenv('EMAIL_USER')
    password = os.getenv('EMAIL_PASSWORD')

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

# Signup function
def signup():
    st.subheader("Sign Up")
    name = st.text_input("Enter your name", key='signup_name')
    email = st.text_input("Enter your email", key='signup_email')
    dob = st.date_input("Enter your date of birth", key='signup_dob')
    pob = st.text_input("Enter your place of birth", key='signup_pob')
    password = st.text_input("Enter a new password", type="password", key='signup_password')
    confirm_password = st.text_input("Confirm your password", type="password", key='signup_confirm_password')

    if st.button("Sign Up"):
        existing_user = session.query(User).filter_by(email=email).first()
        if existing_user:
            st.error("Email already exists. Try a different email.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        elif not password_schema.validate(password):
            st.error("Password does not meet the requirements.")
        else:
            hashed_password = hash_password(password)
            new_user = User(name=name, email=email, password=hashed_password, dob=dob, pob=pob)
            session.add(new_user)
            session.commit()
            st.success("User registered successfully!")

# Login function
def login():
    st.subheader("Login")
    email = st.text_input("Enter your email", key='login_email')
    password = st.text_input("Enter your password", type="password", key='login_password')

    if st.button("Login"):
        try:
            user = session.query(User).filter_by(email=email).one()
            if verify_password(user.password, password):
                st.success("Login successful!")
                st.session_state.logged_in = True
                st.session_state.username = user.name
                st.session_state.email = user.email
                st.session_state.user_id = user.id
            else:
                st.error("Invalid email or password.")
        except NoResultFound:
            st.error("Invalid email or password.")
        except Exception as e:
            st.error(f"Error during login: {e}")

# Forgot password function with security questions
def forgot_password():
    st.subheader("Forgot Password")
    email = st.text_input("Enter your email", key='forgot_email')
    dob = st.date_input("Enter your date of birth", key='forgot_dob')
    pob = st.text_input("Enter your place of birth", key='forgot_pob')

    if 'identity_verified' not in st.session_state:
        st.session_state.identity_verified = False

    if st.button("Submit"):
        try:
            user = session.query(User).filter_by(email=email, dob=dob, pob=pob).one()
            st.session_state.email = email
            st.session_state.user_id = user.id
            st.session_state.identity_verified = True
            st.success("Identity verified. Please reset your password.")
        except NoResultFound:
            st.error("Invalid details provided.")
        except Exception as e:
            st.error(f"Error during password reset: {e}")

    if st.session_state.identity_verified:
        new_password = st.text_input("Enter a new password", type="password", key='reset_new_password')
        confirm_new_password = st.text_input("Confirm your new password", type="password", key='reset_confirm_new_password')

        if st.button("Reset Password"):
            if new_password != confirm_new_password:
                st.error("Passwords do not match.")
            elif not password_schema.validate(new_password):
                st.error("Password does not meet the requirements.")
            else:
                user = session.query(User).filter_by(id=st.session_state.user_id).one()
                user.password = hash_password(new_password)
                session.commit()
                st.success("Password reset successfully. You can now log in with the new password.")
                st.session_state.identity_verified = False

# My Account function to edit details and change password
def my_account():
    st.subheader("My Account")

    if st.session_state.logged_in:
        user = session.query(User).filter_by(id=st.session_state.user_id).one()

        new_name = st.text_input("Update your name", value=user.name, key='account_name')
        new_dob = st.date_input("Update your date of birth", value=user.dob, key='account_dob')
        new_pob = st.text_input("Update your place of birth", value=user.pob, key='account_pob')

        if st.button("Update Details"):
            user.name = new_name
            user.dob = new_dob
            user.pob = new_pob
            session.commit()
            st.success("Details updated successfully!")

        st.subheader("Change Password")
        current_password = st.text_input("Enter your current password", type="password", key='account_current_password')
        new_password = st.text_input("Enter a new password", type="password", key='account_new_password')
        confirm_new_password = st.text_input("Confirm your new password", type="password", key='account_confirm_new_password')

        if st.button("Change Password"):
            if verify_password(user.password, current_password):
                if new_password != confirm_new_password:
                    st.error("Passwords do not match.")
                elif not password_schema.validate(new_password):
                    st.error("Password does not meet the requirements.")
                else:
                    user.password = hash_password(new_password)
                    session.commit()
                    st.success("Password changed successfully!")
            else:
                st.error("Current password is incorrect.")

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.email = ""
    st.session_state.user_id = None

# Main menu function
def main_menu():
    st.subheader("Main Menu")
    menu_options = [f"{st.session_state.username}'s Portfolio",f"{st.session_state.username}'s Watchlist", "Stock Screener",  "Stock Watch", "Technical Analysis", "Stock Prediction",
                    "Stock Comparison", "Market Stats","Markets", "My Account"]
    choice = st.selectbox("Select an option", menu_options)
    return choice

# Sidebar menu
with st.sidebar:
    st.title("TradeSense")
    if st.session_state.logged_in:
        st.write(f"Logged in as: {st.session_state.username}")
        if st.button("Logout"):
            logout()
            st.experimental_rerun()  # Refresh the app
        else:
            choice = main_menu()  # Display the main menu in the sidebar if logged in
    else:
        selected = st.selectbox("Choose an option", ["Login", "Sign Up", "Forgot Password"])
        if selected == "Login":
            login()
        elif selected == "Sign Up":
            signup()
        elif selected == "Forgot Password":
            forgot_password()
        choice = None

# Main content area

def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period='1y')
        if df.empty:
            st.warning(f"No data found for {ticker}.")
            return pd.DataFrame()  # Return an empty DataFrame
        df['2_MA'] = df['Close'].rolling(window=2).mean()
        df['15_MA'] = df['Close'].rolling(window=15).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        return df[['Close', '2_MA', '15_MA', 'RSI', 'ADX']].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.Series(dtype='float64')  # Return an empty Series
    
if not st.session_state.logged_in:
    # dashboard code-------------------------------------------------------------------------------------------------------
    tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO",
                "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS",
                "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS",
                "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS",
                "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS",
                "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO",
                "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS",
                "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ITC.NS", "JBCHEPHARM.BO",
                "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS",
                "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS",
                "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS",
                "PFIZER.NS", "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS",
                "RITES.NS", "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS",
                "SUNTV.NS", "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS",
                "TIMKEN.NS", "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS",
                "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS"]

    # Function to get stock data and calculate moving averages
    @st.cache_data
    def get_stock_data(ticker_symbol, start_date, end_date):
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        data['MA_15'] = data['Close'].rolling(window=15).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data.dropna(inplace=True)
        return data

    # Function to create Plotly figure
    def create_figure(data, indicators, title):
        fig = go.Figure()
        if 'Close' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        if 'MA_15' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_15'], mode='lines', name='15-day MA'))
        if 'MA_50' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-day MA'))
        fig.update_layout(
            title=title, 
            xaxis_title='Date', 
            yaxis_title='Price',
            xaxis_rangeslider_visible=True,
            plot_bgcolor='dark grey',
            paper_bgcolor='white',
            font=dict(color='black'),
            hovermode='x',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            ),
            yaxis=dict(fixedrange=False),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Reset Zoom",
                            method="relayout",
                            args=[{"xaxis.range": [None, None],
                                    "yaxis.range": [None, None]}])]
            )]
        )
        return fig

    # Function to fetch data
    @st.cache_data
    def fetch_data(tickers, period='1d', interval='1m'):
        data = yf.download(tickers, period=period, interval=interval)
        return data['Close']

    # Function to reshape data for heatmap
    def reshape_for_heatmap(df, num_columns=10):
        num_rows = int(np.ceil(len(df) / num_columns))
        reshaped_data = np.zeros((num_rows, num_columns))
        reshaped_tickers = np.empty((num_rows, num_columns), dtype=object)
        reshaped_data[:] = np.nan
        reshaped_tickers[:] = ''
        index = 0
        for y in range(num_rows):
            for x in range(num_columns):
                if index < len(df):
                    reshaped_data[y, x] = df['% Change'].values[index]
                    reshaped_tickers[y, x] = df['Ticker'].values[index]
                    index += 1
        return reshaped_data, reshaped_tickers

    # Create annotated heatmaps using Plotly
    def create_horizontal_annotated_heatmap(df, title, num_columns=10):
        reshaped_data, tickers = reshape_for_heatmap(df, num_columns)
        annotations = []
        for y in range(reshaped_data.shape[0]):
            for x in range(reshaped_data.shape[1]):
                text = f'<b>{tickers[y, x]}</b><br>{reshaped_data[y, x]}%'
                annotations.append(
                    go.layout.Annotation(
                        text=text,
                        x=x,
                        y=y,
                        xref='x',
                        yref='y',
                        showarrow=False,
                        font=dict(size=10, color="black", family="Arial, sans-serif"),
                        align="left"
                    )
                )
        fig = go.Figure(data=go.Heatmap(
            z=reshaped_data,
            x=list(range(reshaped_data.shape[1])),
            y=list(range(reshaped_data.shape[0])),
            hoverinfo='text',
            colorscale='Blues',
            showscale=False,
        ))
        fig.update_layout(
            title=title,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            annotations=annotations,
            autosize=False,
            width=1800,
            height=200 + 50 * len(reshaped_data),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    # Function to fetch stock data and volume
    @st.cache_data
    def get_volume_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data['Volume'].sum()

    # Function to fetch sector data
    @st.cache_data
    def get_sector_data(ticker_symbol, start_date, end_date):
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        return data

    # Function to calculate sector performance
    def calculate_performance(data):
        if not data.empty:
            performance = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            return performance
        return None

    # Function to fetch market data
    @st.cache_data
    def get_market_data(ticker_symbol, start_date, end_date):
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        return data

    st.title("TradeSense")
    st.write("An ultimate platform for smart trading insights. Please log in or sign up to get started.")

    # Create tiles for different sections
    tile_selection = st.selectbox("Select a section", 
                                ["Major Indices", "Top Gainers and Losers", "Volume Chart", 
                                "Sector Performance Chart", "Market Performance"])

    # Major Indices
    if tile_selection == "Major Indices":
        st.subheader("Major Indices")
        col1, col2, col3 = st.columns(3)
        with col1:
            stock_symbols = ["^BSESN", "BSE-500.BO", "^BSEMD", "^BSESMLCAP", "^NSEI", "^NSMIDCP", "^NSEMDCP", "^NSESCP"]
            ticker = st.selectbox("Enter Stock symbol", stock_symbols)
            st.write(f"You selected: {ticker}")
        with col2:
            START = st.date_input('Start Date', pd.to_datetime("2023-06-06"))
        with col3:
            END = st.date_input('End Date', pd.to_datetime("today"))
        if ticker and START and END:
            data = get_stock_data(ticker, START, END)
            fig = create_figure(data, ['Close', 'MA_15', 'MA_50'], f"{ticker} Stock Prices")
            st.plotly_chart(fig)

    # Top Gainers and Losers
    elif tile_selection == "Top Gainers and Losers":
        st.subheader("Top Gainers and Losers")
        
        
        # Fetch data for different periods
        data_daily = fetch_data(tickers, period='1d', interval='1m')
        data_weekly = fetch_data(tickers, period='5d', interval='1d')
        data_monthly = fetch_data(tickers, period='1mo', interval='1d')

        # Clean and prepare data
        data_daily.dropna(axis=1, how='all', inplace=True)
        data_weekly.dropna(axis=1, how='all', inplace=True)
        data_monthly.dropna(axis=1, how='all', inplace=True)
        data_daily.fillna(method='ffill', inplace=True)
        data_weekly.fillna(method='ffill', inplace=True)
        data_monthly.fillna(method='ffill', inplace=True)
        data_daily.fillna(method='bfill', inplace=True)
        data_weekly.fillna(method='bfill', inplace=True)
        data_monthly.fillna(method='bfill', inplace=True)

        # Calculate changes
        daily_change = data_daily.iloc[-1] - data_daily.iloc[0]
        percent_change_daily = (daily_change / data_daily.iloc[0]) * 100
        weekly_change = data_weekly.iloc[-1] - data_weekly.iloc[0]
        percent_change_weekly = (weekly_change / data_weekly.iloc[0]) * 100
        monthly_change = data_monthly.iloc[-1] - data_monthly.iloc[0]
        percent_change_monthly = (monthly_change / data_monthly.iloc[0]) * 100

        # Create DataFrames
        df_daily = pd.DataFrame({'Ticker': data_daily.columns, 'Last Traded Price': data_daily.iloc[-1].values, '% Change': percent_change_daily.values})
        df_weekly = pd.DataFrame({'Ticker': data_weekly.columns, 'Last Traded Price': data_weekly.iloc[-1].values, '% Change': percent_change_weekly.values})
        df_monthly = pd.DataFrame({'Ticker': data_monthly.columns, 'Last Traded Price': data_monthly.iloc[-1].values, '% Change': percent_change_monthly.values})

        # Round off the % Change values and sort
        df_daily['% Change'] = df_daily['% Change'].round(2)
        df_weekly['% Change'] = df_weekly['% Change'].round(2)
        df_monthly['% Change'] = df_monthly['% Change'].round(2)
        df_daily_sorted = df_daily.sort_values(by='% Change', ascending=True)
        df_weekly_sorted = df_weekly.sort_values(by='% Change', ascending=True)
        df_monthly_sorted = df_monthly.sort_values(by='% Change', ascending=True)

        # Dropdown menu to select the period
        heatmap_option = st.selectbox('Select to view:', ['Daily Gainers/Losers', 'Weekly Gainers/Losers', 'Monthly Gainers/Losers'])

        # Display the selected heatmap
        if heatmap_option == 'Daily Gainers/Losers':
            fig = create_horizontal_annotated_heatmap(df_daily_sorted, 'Daily Gainers/Losers')
            st.plotly_chart(fig)
        elif heatmap_option == 'Weekly Gainers/Losers':
            fig = create_horizontal_annotated_heatmap(df_weekly_sorted, 'Weekly Gainers/Losers')
            st.plotly_chart(fig)
        elif heatmap_option == 'Monthly Gainers/Losers':
            fig = create_horizontal_annotated_heatmap(df_monthly_sorted, 'Monthly Gainers/Losers')
            st.plotly_chart(fig)

    # Volume Chart
    elif tile_selection == "Volume Chart":
        st.subheader("Volume Chart")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', datetime(2022, 1, 1), key='start_date')
        with col2:
            end_date = st.date_input('End Date', datetime.today(), key='end_date')
        volume_data = {ticker: get_volume_data(ticker, start_date, end_date) for ticker in tickers}
        volume_df = pd.DataFrame(list(volume_data.items()), columns=['Ticker', 'Volume'])
        fig = px.bar(volume_df, x='Ticker', y='Volume', title='Trading Volume of Stocks',
                    labels={'Volume': 'Total Volume'}, color='Volume',
                    color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

    # Sector Performance Chart
    elif tile_selection == "Sector Performance Chart":
        st.subheader("Sector Performance Chart")
        sector_indices = {
            'NIFTY_BANK': '^NSEBANK',
            'NIFTY_IT': '^CNXIT',
            'NIFTY_AUTO': '^CNXAUTO',
            'NIFTY_FMCG': '^CNXFMCG',
            'NIFTY_PHARMA': '^CNXPHARMA',
            'NIFTY_REALTY': '^CNXREALTY',
            'NIFTY_METAL': '^CNXMETAL',
            'NIFTY_MEDIA': '^CNXMEDIA',
            'NIFTY_PSU_BANK': '^CNXPSUBANK',
            'NIFTY_ENERGY': '^CNXENERGY',
            'NIFTY_COMMODITIES': '^CNXCOMMOD',
            'NIFTY_INFRASTRUCTURE': '^CNXINFRA',
            'NIFTY_SERVICES_SECTOR': '^CNXSERVICE',
            'NIFTY_FINANCIAL_SERVICES': '^CNXFINANCE',
            'NIFTY_MNC': '^CNXMNC',
            'NIFTY_PSE': '^CNXPSE',
            'NIFTY_CPSE': '^CNXCPSE',
            'NIFTY_100': '^CNX100',
            'NIFTY_200': '^CNX200',
            'NIFTY_500': '^CNX500',
            'NIFTY_MIDCAP_50': '^CNXMID50',
            'NIFTY_MIDCAP_100': '^CNXMIDCAP',
            'NIFTY_SMALLCAP_100': '^CNXSMCAP',
            'NIFTY_NEXT_50': '^CNXNIFTY'
        }
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', datetime(2022, 1, 1), key='start_date')
        with col2:
            end_date = st.date_input('End Date', datetime.today(), key='end_date')
        sector_performance = {sector: calculate_performance(get_sector_data(ticker, start_date, end_date)) for sector, ticker in sector_indices.items() if calculate_performance(get_sector_data(ticker, start_date, end_date)) is not None}
        performance_df = pd.DataFrame(list(sector_performance.items()), columns=['Sector', 'Performance'])
        fig = px.bar(performance_df, x='Sector', y='Performance', title='Sector Performance',
                    labels={'Performance': 'Performance (%)'}, color='Performance',
                    color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

    # Market Performance
    elif tile_selection == "Market Performance":
        st.subheader("Market Performance")
        market_indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'Gold': 'GC=F',
            'Silver': 'SI=F',
            'Oil': 'CL=F',
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD'
        }
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', datetime(2022, 1, 1), key='start_date')
        with col2:
            end_date = st.date_input('End Date', datetime.today(), key='end_date')
        market_performance = {market: calculate_performance(get_market_data(ticker, start_date, end_date)) for market, ticker in market_indices.items() if calculate_performance(get_market_data(ticker, start_date, end_date)) is not None}
        performance_df = pd.DataFrame(list(market_performance.items()), columns=['Market', 'Performance'])
        fig = px.bar(performance_df, x='Market', y='Performance', title='Market Performance',
                    labels={'Performance': 'Performance (%)'}, color='Performance',
                    color_continuous_scale=px.colors.diverging.RdYlGn)
        st.plotly_chart(fig)

    st.markdown("-----------------------------------------------------------------------------------------------------------------------")
    st.subheader("Unlock your trading potential. Join TradeSense today!")
    

else:
    if choice:
        if choice == "My Account":
            my_account()
        elif choice == f"{st.session_state.username}'s Watchlist":
            # 'watchlist' code ---------------------------------------------------------------------------------------------------------------------------------------------------------------
            st.header(f"{st.session_state.username}'s Watchlist")
            user_id = session.query(User.id).filter_by(email=st.session_state.email).first()[0]
            watchlist = session.query(Watchlist).filter_by(user_id=user_id).all()

            def fetch_ticker_data(ticker):
                try:
                    stock = yf.Ticker(ticker)
                    data = stock.history(period="1y")
                    if data.empty:
                        raise ValueError("Ticker not found")
                    return data
                except Exception as e:
                    raise ValueError("Ticker not found") from e

            def calculate_indicators(data):
                try:
                    data['5_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=5)
                    data['15_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=15)
                    data['MACD'] = ta.trend.macd(data['Close'])
                    data['MACD_Hist'] = ta.trend.macd_diff(data['Close'])
                    data['RSI'] = ta.momentum.rsi(data['Close'])
                    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
                    data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'])
                    data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'])
                    data['20_day_vol_MA'] = data['Volume'].rolling(window=20).mean()
                    return data
                except Exception as e:
                    raise ValueError("Error calculating indicators") from e

            if st.session_state.username:
                choice = f"{st.session_state.username}'s Watchlist"
                if choice == f"{st.session_state.username}'s Watchlist":
                    st.header(f"{st.session_state.username}'s Watchlist")
                    user_id = session.query(User.id).filter_by(email=st.session_state.email).first()[0]
                    watchlist = session.query(Watchlist).filter_by(user_id=user_id).all()

                    # Add new ticker to watchlist
                    new_ticker = st.text_input("Add a new ticker to your watchlist")
                    if st.button("Add Ticker"):
                        try:
                            fetch_ticker_data(new_ticker)
                            if not session.query(Watchlist).filter_by(user_id=user_id, ticker=new_ticker).first():
                                new_watchlist_entry = Watchlist(user_id=user_id, ticker=new_ticker)
                                session.add(new_watchlist_entry)
                                session.commit()
                                st.success(f"{new_ticker} added to your watchlist!")
                                # Refresh watchlist data
                                watchlist = session.query(Watchlist).filter_by(user_id=user_id).all()
                            else:
                                st.warning(f"{new_ticker} is already in your watchlist.")
                        except ValueError as ve:
                            st.error(ve)

                    # Display watchlist
                    if watchlist:
                        watchlist_data = {}
                        for entry in watchlist:
                            ticker = entry.ticker
                            try:
                                data = fetch_ticker_data(ticker)
                                data = calculate_indicators(data)
                                latest_data = data.iloc[-1]
                                watchlist_data[ticker] = {
                                    'Close': latest_data['Close'],
                                    '5_day_EMA': latest_data['5_day_EMA'],
                                    '15_day_EMA': latest_data['15_day_EMA'],
                                    'MACD': latest_data['MACD'],
                                    'MACD_Hist': latest_data['MACD_Hist'],
                                    'RSI': latest_data['RSI'],
                                    'ADX': latest_data['ADX'],
                                    'Bollinger_High': latest_data['Bollinger_High'],
                                    'Bollinger_Low': latest_data['Bollinger_Low'],
                                    'Volume': latest_data['Volume'],
                                    '20_day_vol_MA': latest_data['20_day_vol_MA']
                                }
                            except ValueError as ve:
                                st.error(f"Error fetching data for {ticker}: {ve}")

                        watchlist_df = pd.DataFrame.from_dict(watchlist_data, orient='index')
                        st.write("Your Watchlist:")
                        st.dataframe(watchlist_df)

                        # Option to remove ticker from watchlist
                        ticker_to_remove = st.selectbox("Select a ticker to remove", [entry.ticker for entry in watchlist])
                        if st.button("Remove Ticker"):
                            session.query(Watchlist).filter_by(user_id=user_id, ticker=ticker_to_remove).delete()
                            session.commit()
                            st.success(f"{ticker_to_remove} removed from your watchlist.")
                            st.experimental_rerun()  # Refresh the app to reflect changes
                    else:
                        st.write("Your watchlist is empty.")

        elif choice == f"{st.session_state.username}'s Portfolio":
            # 'Portifolio' code -------------------------------------------------------------------------------------------------------------------------------------------------------
            st.header(f"{st.session_state.username}'s Portfolio")
            user_id = session.query(User.id).filter_by(email=st.session_state.email).first()[0]
            portfolio = session.query(Portfolio).filter_by(user_id=user_id).all()

            # Add new stock to portfolio
            st.subheader("Add to Portfolio")
            # Create three columns
            col1, col2, col3 = st.columns(3)
            with col1:
                new_ticker = st.text_input("Ticker Symbol")
            with col2:
                shares = st.number_input("Number of Shares", min_value=0.0, step=0.01)
            with col3:
                bought_price = st.number_input("Bought Price per Share", min_value=0.0, step=0.01)
            if st.button("Add to Portfolio"):
                if not session.query(Portfolio).filter_by(user_id=user_id, ticker=new_ticker).first():
                    new_portfolio_entry = Portfolio(user_id=user_id, ticker=new_ticker, shares=shares,
                                                    bought_price=bought_price)
                    session.add(new_portfolio_entry)
                    session.commit()
                    st.success(f"{new_ticker} added to your portfolio!")
                    # Refresh portfolio data
                    portfolio = session.query(Portfolio).filter_by(user_id=user_id).all()
                else:
                    st.warning(f"{new_ticker} is already in your portfolio.")

            # Display portfolio
            if portfolio:
                portfolio_data = []
                invested_values = []
                current_values = []
                for entry in portfolio:
                    current_data = yf.download(entry.ticker, period='1d')
                    last_price = current_data['Close'].iloc[-1]
                    invested_value = entry.shares * entry.bought_price
                    current_value = entry.shares * last_price
                    p_l = current_value - invested_value
                    p_l_percent = (p_l / invested_value) * 100
                    portfolio_data.append({
                        "Ticker": entry.ticker,
                        "Shares": entry.shares,
                        "Bought Price": entry.bought_price,
                        "Invested Value": invested_value,
                        "Last Traded Price": last_price,
                        "Current Value": current_value,
                        "P&L (%)": p_l_percent
                    })
                    invested_values.append(invested_value)
                    current_values.append(current_value)
                portfolio_df = pd.DataFrame(portfolio_data)
                
                
                st.write("Your Portfolio:")
                st.dataframe(portfolio_df)

                # Create two columns for date input
                col1, col2 = st.columns(2)

                # Set up the start and end date inputs with default values

                with col1:
                    # Generate donut chart
                    labels = portfolio_df['Ticker']
                    values = portfolio_df['Current Value']
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
                    fig.update_layout(title_text="Portfolio Distribution")
                    st.plotly_chart(fig)

                with col2:
  
                    # Generate histogram for sum of Invested Value and Current Value
                    total_invested_value = sum(invested_values)
                    total_current_value = sum(current_values)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=['Total Invested Value', 'Total Current Value'], y=[total_invested_value, total_current_value]))
                    fig.update_layout(title_text='Profit/Loss',
                                xaxis_title='Value', yaxis_title='Sum')
                    st.plotly_chart(fig)

                # Option to remove stock from portfolio
                ticker_to_remove = st.selectbox("Select a ticker to remove", [entry.ticker for entry in portfolio])
                if st.button("Remove from Portfolio"):
                    session.query(Portfolio).filter_by(user_id=user_id, ticker=ticker_to_remove).delete()
                    session.commit()
                    st.success(f"{ticker_to_remove} removed from your portfolio.")
                    st.experimental_rerun()  # Refresh the app to reflect changes
            else:
                st.write("Your portfolio is empty.")

        elif choice == "Markets":
        
                #'Markets' code-------------------------------------------------------------------------------------------------------------------------------------------------------
                # Function to download data and calculate moving averages with caching
                @lru_cache(maxsize=32)
                def get_stock_data(ticker_symbol, start_date, end_date):
                    data = yf.download(ticker_symbol, start=start_date, end=end_date)
                    data['MA_15'] = data['Close'].rolling(window=15).mean()
                    data['MA_50'] = data['Close'].rolling(window=50).mean()
                    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
                    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
                    data['Upper_Band'] = data['Close'].rolling(20).mean() + (data['Close'].rolling(20).std() * 2)
                    data['Lower_Band'] = data['Close'].rolling(20).mean() - (data['Close'].rolling(20).std() * 2)
                    data.dropna(inplace=True)
                    return data

                # Function to create Plotly figure
                def create_figure(data, indicators, title):
                    fig = go.Figure()
                    if 'Close' in indicators:
                        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
                    if 'MA_15' in indicators:
                        fig.add_trace(go.Scatter(x=data.index, y=data['MA_15'], mode='lines', name='15-day MA'))
                    if 'MA_50' in indicators:
                        fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-day MA'))
                    if 'MACD' in indicators:
                        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
                        fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line'))
                    if 'Bollinger Bands' in indicators:
                        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], mode='lines', name='Upper Band'))
                        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], mode='lines', name='Lower Band'))

                    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price',
                                    xaxis_rangeslider_visible=True,
                                    plot_bgcolor='dark grey',
                                    paper_bgcolor='white',
                                    font=dict(color='black'),
                                    hovermode='x',
                                    xaxis=dict(rangeselector=dict(buttons=list([
                                        dict(count=1, label="1m", step="month", stepmode="backward"),
                                        dict(count=6, label="6m", step="month", stepmode="backward"),
                                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                                        dict(count=1, label="1y", step="year", stepmode="backward"),
                                        dict(step="all")
                                    ])),
                                        rangeslider=dict(visible=True),
                                        type='date'),
                                    yaxis=dict(fixedrange=False),
                                    updatemenus=[dict(type="buttons",
                                                        buttons=[dict(label="Reset Zoom",
                                                                    method="relayout",
                                                                    args=[{"xaxis.range": [None, None],
                                                                            "yaxis.range": [None, None]}])])])
                    return fig

                # Function to calculate correlation
                def calculate_correlation(data1, data2):
                    return data1['Close'].corr(data2['Close'])

                # Function to plot correlation matrix
                def plot_correlation_matrix(correlation_matrix):
                    fig = go.Figure(data=go.Heatmap(
                        z=correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        colorscale='Viridis'))
                    fig.update_layout(title="Correlation Matrix", xaxis_title='Assets', yaxis_title='Assets')
                    return fig

                # Function to calculate Sharpe Ratio
                def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
                    return (returns.mean() - risk_free_rate) / returns.std()

                # Function to calculate Beta
                def calculate_beta(asset_returns, market_returns):
                    # Align the series to have the same index
                    aligned_returns = pd.concat([asset_returns, market_returns], axis=1).dropna()
                    covariance_matrix = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])
                    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
                    return beta

                # Function to calculate Value at Risk (VaR)
                def calculate_var(returns, confidence_level=0.05):
                    return np.percentile(returns, confidence_level * 100)

                # Main application
                st.title("Market Insights")

                # Date inputs
                col1, col2 = st.columns(2)
                with col1:
                    START = st.date_input('Start Date', pd.to_datetime("2023-06-06"))
                with col2:
                    END = st.date_input('End Date', pd.to_datetime("today"))

                # Markets submenu
                submenu = st.sidebar.radio("Select Option", ["Equities", "Commodities", "Currencies", "Cryptocurrencies", "Insights"])

                if submenu == "Equities":
                    st.subheader("Equity Markets")
                    data_nyse = get_stock_data("^NYA", START, END)
                    data_bse = get_stock_data("^BSESN", START, END)
                    indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
                    fig_nyse = create_figure(data_nyse, indicators, 'NYSE Price')
                    fig_bse = create_figure(data_bse, indicators, 'BSE Price')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_nyse)

                    with col2:
                        st.plotly_chart(fig_bse)


                elif submenu == "Commodities":
                    st.subheader("Commodities")
                    tickers = ["GC=F", "CL=F", "NG=F", "SI=F", "HG=F"]
                    selected_tickers = st.multiselect("Select stock tickers to visualize", tickers, default=["GC=F", "CL=F"])
                    indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
                    if not selected_tickers:
                        st.warning("Please select at least one ticker.")
                    else:
                        columns = st.columns(len(selected_tickers))
                        for ticker, col in zip(selected_tickers, columns):
                            data = get_stock_data(ticker, START, END)
                            fig = create_figure(data, indicators, f'{ticker} Price')
                            col.plotly_chart(fig)

                elif submenu == "Currencies":
                    st.subheader("Currencies")
                    tickers = ["EURUSD=X", "GBPUSD=X", "CNYUSD=X", "INRUSD=X"]
                    selected_tickers = st.multiselect("Select currency pairs to visualize", tickers, default=["INRUSD=X", "CNYUSD=X"])
                    indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
                    if not selected_tickers:
                        st.warning("Please select at least one currency pair.")
                    else:
                        columns = st.columns(len(selected_tickers))
                        for ticker, col in zip(selected_tickers, columns):
                            data = get_stock_data(ticker, START, END)
                            fig = create_figure(data, indicators, f'{ticker} Price')
                            col.plotly_chart(fig)

                elif submenu == "Cryptocurrencies":
                    st.subheader("Cryptocurrencies")
                    tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
                    selected_tickers = st.multiselect("Select cryptocurrencies to visualize", tickers, default=["BTC-USD", "ETH-USD"])
                    indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
                    if not selected_tickers:
                        st.warning("Please select at least one cryptocurrency.")
                    else:
                        columns = st.columns(len(selected_tickers))
                        for ticker, col in zip(selected_tickers, columns):
                            data = get_stock_data(ticker, START, END)
                            fig = create_figure(data, indicators, f'{ticker} Price')
                            col.plotly_chart(fig)

                elif submenu == "Insights":
                    st.subheader("Detailed Market Analysis")
                    st.write("This section provides an in-depth analysis of the markets, commodities, forex, and cryptos.")

                    # Get data for all categories
                    data_nyse = get_stock_data("^NYA", START, END)
                    data_bse = get_stock_data("^BSESN", START, END)
                    data_gold = get_stock_data("GC=F", START, END)
                    data_oil = get_stock_data("CL=F", START, END)
                    data_eurusd = get_stock_data("EURUSD=X", START, END)
                    data_gbpusd = get_stock_data("GBPUSD=X", START, END)
                    data_btc = get_stock_data("BTC-USD", START, END)
                    data_eth = get_stock_data("ETH-USD", START, END)

                    # Calculate correlations
                    correlation_data = {
                        'NYSE': data_nyse['Close'],
                        'BSE': data_bse['Close'],
                        'Gold': data_gold['Close'],
                        'Oil': data_oil['Close'],
                        'EURUSD': data_eurusd['Close'],
                        'GBPUSD': data_gbpusd['Close'],
                        'BTC': data_btc['Close'],
                        'ETH': data_eth['Close']
                    }
                    df_correlation = pd.DataFrame(correlation_data)
                    correlation_matrix = df_correlation.corr()

                    # Plot correlation matrix
                    fig_corr_matrix = plot_correlation_matrix(correlation_matrix)
                    st.plotly_chart(fig_corr_matrix)

                    # Calculate market returns for beta calculation (assuming S&P 500 as market index)
                    data_sp500 = get_stock_data("^GSPC", START, END)
                    market_returns = data_sp500['Close'].pct_change().dropna()

                    # Trend and Additional Insights Analysis
                    st.write("**Trend Analysis and Insights:**")
                    analysis_data = {
                        "Assets": ['NYSE', 'BSE', 'Gold', 'Oil', 'EURUSD', 'GBPUSD', 'BTC', 'ETH'],
                        "Trend": [
                            "Bullish" if data_nyse['MA_15'].iloc[-1] > data_nyse['MA_50'].iloc[-1] else "Bearish",
                            "Bullish" if data_bse['MA_15'].iloc[-1] > data_bse['MA_50'].iloc[-1] else "Bearish",
                            "Bullish" if data_gold['MA_15'].iloc[-1] > data_gold['MA_50'].iloc[-1] else "Bearish",
                            "Bullish" if data_oil['MA_15'].iloc[-1] > data_oil['MA_50'].iloc[-1] else "Bearish",
                            "Bullish" if data_eurusd['MA_15'].iloc[-1] > data_eurusd['MA_50'].iloc[-1] else "Bearish",
                            "Bullish" if data_gbpusd['MA_15'].iloc[-1] > data_gbpusd['MA_50'].iloc[-1] else "Bearish",
                            "Bullish" if data_btc['MA_15'].iloc[-1] > data_btc['MA_50'].iloc[-1] else "Bearish",
                            "Bullish" if data_eth['MA_15'].iloc[-1] > data_eth['MA_50'].iloc[-1] else "Bearish"
                        ],
                        "Volatility (Daily)": [
                            np.std(data_nyse['Close']),
                            np.std(data_bse['Close']),
                            np.std(data_gold['Close']),
                            np.std(data_oil['Close']),
                            np.std(data_eurusd['Close']),
                            np.std(data_gbpusd['Close']),
                            np.std(data_btc['Close']),
                            np.std(data_eth['Close'])
                        ],
                        "Average Return (%) (Daily)": [
                            np.mean(data_nyse['Close'].pct_change()) * 100,
                            np.mean(data_bse['Close'].pct_change()) * 100,
                            np.mean(data_gold['Close'].pct_change()) * 100,
                            np.mean(data_oil['Close'].pct_change()) * 100,
                            np.mean(data_eurusd['Close'].pct_change()) * 100,
                            np.mean(data_gbpusd['Close'].pct_change()) * 100,
                            np.mean(data_btc['Close'].pct_change()) * 100,
                            np.mean(data_eth['Close'].pct_change()) * 100
                        ],
                        "Sharpe Ratio (Daily)": [
                            calculate_sharpe_ratio(data_nyse['Close'].pct_change()),
                            calculate_sharpe_ratio(data_bse['Close'].pct_change()),
                            calculate_sharpe_ratio(data_gold['Close'].pct_change()),
                            calculate_sharpe_ratio(data_oil['Close'].pct_change()),
                            calculate_sharpe_ratio(data_eurusd['Close'].pct_change()),
                            calculate_sharpe_ratio(data_gbpusd['Close'].pct_change()),
                            calculate_sharpe_ratio(data_btc['Close'].pct_change()),
                            calculate_sharpe_ratio(data_eth['Close'].pct_change())
                        ],
                        "Max Drawdown (%)": [
                            (data_nyse['Close'].max() - data_nyse['Close'].min()) / data_nyse['Close'].max() * 100,
                            (data_bse['Close'].max() - data_bse['Close'].min()) / data_bse['Close'].max() * 100,
                            (data_gold['Close'].max() - data_gold['Close'].min()) / data_gold['Close'].max() * 100,
                            (data_oil['Close'].max() - data_oil['Close'].min()) / data_oil['Close'].max() * 100,
                            (data_eurusd['Close'].max() - data_eurusd['Close'].min()) / data_eurusd['Close'].max() * 100,
                            (data_gbpusd['Close'].max() - data_gbpusd['Close'].min()) / data_gbpusd['Close'].max() * 100,
                            (data_btc['Close'].max() - data_btc['Close'].min()) / data_btc['Close'].max() * 100,
                            (data_eth['Close'].max() - data_eth['Close'].min()) / data_eth['Close'].max() * 100
                        ],
                        "Beta": [
                            calculate_beta(data_nyse['Close'].pct_change().dropna(), market_returns),
                            calculate_beta(data_bse['Close'].pct_change().dropna(), market_returns),
                            calculate_beta(data_gold['Close'].pct_change().dropna(), market_returns),
                            calculate_beta(data_oil['Close'].pct_change().dropna(), market_returns),
                            calculate_beta(data_eurusd['Close'].pct_change().dropna(), market_returns),
                            calculate_beta(data_gbpusd['Close'].pct_change().dropna(), market_returns),
                            calculate_beta(data_btc['Close'].pct_change().dropna(), market_returns),
                            calculate_beta(data_eth['Close'].pct_change().dropna(), market_returns)
                        ],
                        "Value at Risk (VaR) 5%": [
                            calculate_var(data_nyse['Close'].pct_change().dropna()),
                            calculate_var(data_bse['Close'].pct_change().dropna()),
                            calculate_var(data_gold['Close'].pct_change().dropna()),
                            calculate_var(data_oil['Close'].pct_change().dropna()),
                            calculate_var(data_eurusd['Close'].pct_change().dropna()),
                            calculate_var(data_gbpusd['Close'].pct_change().dropna()),
                            calculate_var(data_btc['Close'].pct_change().dropna()),
                            calculate_var(data_eth['Close'].pct_change().dropna())
                        ]
                    }
                    df_analysis = pd.DataFrame(analysis_data)
                    st.table(df_analysis)

                    # Annualized metrics
                    st.write("**Annualized Metrics:**")
                    annualized_data = {
                        "Assets": ['NYSE', 'BSE', 'Gold', 'Oil', 'EURUSD', 'GBPUSD', 'BTC', 'ETH'],
                        "Annualized Return (%)": [
                            ((1 + np.mean(data_nyse['Close'].pct_change())) ** 252 - 1) * 100,
                            ((1 + np.mean(data_bse['Close'].pct_change())) ** 252 - 1) * 100,
                            ((1 + np.mean(data_gold['Close'].pct_change())) ** 252 - 1) * 100,
                            ((1 + np.mean(data_oil['Close'].pct_change())) ** 252 - 1) * 100,
                            ((1 + np.mean(data_eurusd['Close'].pct_change())) ** 252 - 1) * 100,
                            ((1 + np.mean(data_gbpusd['Close'].pct_change())) ** 252 - 1) * 100,
                            ((1 + np.mean(data_btc['Close'].pct_change())) ** 252 - 1) * 100,
                            ((1 + np.mean(data_eth['Close'].pct_change())) ** 252 - 1) * 100
                        ],
                        "Annualized Volatility (%)": [
                            np.std(data_nyse['Close'].pct_change()) * np.sqrt(252) * 100,
                            np.std(data_bse['Close'].pct_change()) * np.sqrt(252) * 100,
                            np.std(data_gold['Close'].pct_change()) * np.sqrt(252) * 100,
                            np.std(data_oil['Close'].pct_change()) * np.sqrt(252) * 100,
                            np.std(data_eurusd['Close'].pct_change()) * np.sqrt(252) * 100,
                            np.std(data_gbpusd['Close'].pct_change()) * np.sqrt(252) * 100,
                            np.std(data_btc['Close'].pct_change()) * np.sqrt(252) * 100,
                            np.std(data_eth['Close'].pct_change()) * np.sqrt(252) * 100
                        ]
                    }
                    df_annualized = pd.DataFrame(annualized_data)
                    st.table(df_annualized)

        elif choice == "Stock Screener":
            # 'Stock Screener' code--------------------------------------------------------------------------------------------------------------------------------------------------------
            st.sidebar.subheader("Indices")
            submenu = st.sidebar.radio("Select Option", ["MAD-LargeCap", "MAD-MidCap", "BSE-LargeCap","BSE-MidCap","BSE-SmallCap","FTSE100","S&P500"])

            # Function to create Plotly figure
            def create_figure(data, indicators, title):
                fig = go.Figure()

                fig.update_layout(
                    title=title, 
                    xaxis_title='Date', 
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=True,
                    plot_bgcolor='dark grey',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    hovermode='x',
                    xaxis=dict(
                        rangeselector=dict(),
                        rangeslider=dict(visible=True),
                        type='date'
                    ),
                    yaxis=dict(fixedrange=False),
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(label="Reset Zoom",
                                    method="relayout",
                                    args=[{"xaxis.range": [None, None],
                                            "yaxis.range": [None, None]}])]
                    )]
                )
                return fig

            # Function to fetch and process stock data
            @st.cache_data(ttl=3600)
            def get_stock_data(ticker_symbols, period):
                try:
                    stock_data = {}
                    for ticker_symbol in ticker_symbols:
                        df = yf.download(ticker_symbol, period=period)
                        if not df.empty:
                            df.interpolate(method='linear', inplace=True)
                            df = calculate_indicators(df)
                            df.dropna(inplace=True)
                            stock_data[ticker_symbol] = df
                    return stock_data
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    return {}

            # Function to calculate technical indicators
            @st.cache_data(ttl=3600)
            def calculate_indicators(df):
                # Calculate Moving Averages
                df['5_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=5).wma()
                df['20_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=20).wma()
                df['50_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=50).wma()

                # Calculate MACD
                macd = ta.trend.MACD(df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_Signal'] = macd.macd_signal()
                df['MACD_Histogram'] = macd.macd_diff()

                # Calculate ADX
                adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
                df['ADX'] = adx.adx()

                # Calculate Parabolic SAR
                psar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'])
                df['Parabolic_SAR'] = psar.psar()

                # Calculate RSI
                rsi = ta.momentum.RSIIndicator(df['Close'])
                df['RSI'] = rsi.rsi()

                # Calculate Volume Moving Averages
                df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
                df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
                df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

                # Calculate Bollinger Bands
                bollinger = ta.volatility.BollingerBands(df['Close'])
                df['Bollinger_High'] = bollinger.bollinger_hband()
                df['Bollinger_Low'] = bollinger.bollinger_lband()
                df['Bollinger_Middle'] = bollinger.bollinger_mavg()

                # Calculate Detrended Price Oscillator (DPO)
                df['DPO'] = ta.trend.DPOIndicator(close=df['Close']).dpo()

                # Calculate On-Balance Volume (OBV)
                df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()

                # Calculate Volume Weighted Average Price (VWAP)
                df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()

                # Calculate Accumulation/Distribution Line (A/D Line)
                df['A/D Line'] = ta.volume.AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).acc_dist_index()

                # Calculate Average True Range (ATR)
                df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()

                return df

            # Function to query the stocks
            @st.cache_data(ttl=3600)
            def query_stocks(stock_data, conditions):
                results = []
                for ticker, df in stock_data.items():
                    if df.empty or len(df) < 1:
                        continue
                    condition_met = True
                    for condition in conditions:
                        col1, op, col2 = condition
                        if col1 not in df.columns or col2 not in df.columns:
                            condition_met = False
                            break
                        if op == '>':
                            if not (df[col1] > df[col2]).iloc[-1]:
                                condition_met = False
                                break
                        elif op == '<':
                            if not (df[col1] < df[col2]).iloc[-1]:
                                condition_met = False
                                break
                        elif op == '>=':
                            if not (df[col1] >= df[col2]).iloc[-1]:
                                condition_met = False
                                break
                        elif op == '<=':
                            if not (df[col1] <= df[col2]).iloc[-1]:
                                condition_met = False
                                break
                    if condition_met:
                        row = {
                            'Ticker': ticker,
                            'MACD': df['MACD'].iloc[-1],
                            'MACD_Signal': df['MACD_Signal'].iloc[-1],
                            'MACD_Hist': df['MACD_Histogram'].iloc[-1],
                            'RSI': df['RSI'].iloc[-1],
                            'ADX': df['ADX'].iloc[-1],
                            'Close': df['Close'].iloc[-1],
                            '5_MA': df['5_MA'].iloc[-1],
                            '20_MA': df['20_MA'].iloc[-1],
                            'Bollinger_High': df['Bollinger_High'].iloc[-1],
                            'Bollinger_Low': df['Bollinger_Low'].iloc[-1],
                            'Bollinger_Middle': df['Bollinger_Middle'].iloc[-1],
                            'Parabolic_SAR': df['Parabolic_SAR'].iloc[-1],
                            'Volume': df['Volume'].iloc[-1],
                            'Volume_MA_10': df['Volume_MA_10'].iloc[-1],
                            'Volume_MA_20': df['Volume_MA_20'].iloc[-1],
                            'DPO': df['DPO'].iloc[-1]
                        }
                        results.append(row)
                return pd.DataFrame(results)

            # Determine tickers based on submenu selection
            if submenu == "MAD-LargeCap":
                st.subheader("MAD-LargeCap")
                tickers = largecap_tickers
            elif submenu == "MAD-MidCap":
                st.subheader("MAD-MidCap")
                tickers = midcap_tickers

            elif submenu == "BSE-LargeCap":
                st.subheader("MAD-LargeCap")
                tickers = bse_largecap

            elif submenu == "BSE-MidCap":
                st.subheader("BSE-MidCap")
                tickers = bse_midcap
            elif submenu == "BSE-SmallCap":
                st.subheader("BSE-SmallCap")
                tickers = bse_smallcap

            elif submenu == "FTSE100":
                st.subheader("FTSE100")
                tickers = ftse100_tickers
            elif submenu == "S&P500":
                st.subheader("S&P500")
                tickers = sp500_tickers

            
            # Fetch data and calculate indicators for each stock
            stock_data = get_stock_data(tickers, period='6mo')

            # Define first set of conditions
            first_conditions = [
                ('MACD', '>', 'MACD_Signal'),
                ('Parabolic_SAR', '<', 'Close')

            ]

            # Query stocks based on the first set of conditions
            first_query_df = query_stocks(stock_data, first_conditions)

            # Filter stocks in an uptrend with high volume and positive DPO
            second_query_df = first_query_df[
                (first_query_df['RSI'] < 67) & (first_query_df['RSI'] > 45) & 
                (first_query_df['ADX'] > 25) & (first_query_df['MACD'] > 0)
            ]

            st.write("Stocks in an uptrend with high volume and positive DPO:")
            st.dataframe(second_query_df)

            # Dropdown for analysis type
            col1, col2 = st.columns(2)

            # Set up the start and end date inputs
            with col1:
                selected_stock = st.selectbox("Select Stock", second_query_df['Ticker'].tolist())

            with col2:
                analysis_type = st.selectbox("Select Analysis Type", ["Trend Analysis", "Volume Analysis", "Support & Resistance Levels"])

            # Create two columns
            col1, col2 = st.columns(2)

            # Set up the start and end date inputs
            with col1:
                START = st.date_input('Start Date', pd.to_datetime("2023-06-01"))

            with col2:
                END = st.date_input('End Date', pd.to_datetime("today"))

            # If a stock is selected, plot its data with the selected indicators
            if selected_stock:
                @st.cache_data(ttl=3600)
                def load_data(ticker, start, end):
                    df = yf.download(ticker, start=start, end=end)
                    df.reset_index(inplace=True)
                    return df

                df = load_data(selected_stock, START, END)

                if df.empty:
                    st.write("No data available for the provided ticker.")
                else:
                    df.interpolate(method='linear', inplace=True)
                    df = calculate_indicators(df)

                    # Identify Horizontal Support and Resistance
                    def find_support_resistance(df, window=20):
                        df['Support'] = df['Low'].rolling(window, center=True).min()
                        df['Resistance'] = df['High'].rolling(window, center=True).max()
                        return df

                    df = find_support_resistance(df)

                    # Draw Trendlines
                    def calculate_trendline(df, kind='support'):
                        if kind == 'support':
                            prices = df['Low']
                        elif kind == 'resistance':
                            prices = df['High']
                        else:
                            raise ValueError("kind must be either 'support' or 'resistance'")

                        indices = np.arange(len(prices))
                        slope, intercept, _, _, _ = linregress(indices, prices)
                        trendline = slope * indices + intercept
                        return trendline

                    df['Support_Trendline'] = calculate_trendline(df, kind='support')
                    df['Resistance_Trendline'] = calculate_trendline(df, kind='resistance')

                    # Calculate Fibonacci Retracement Levels
                    def fibonacci_retracement_levels(high, low):
                        diff = high - low
                        levels = {
                            'Level_0': high,
                            'Level_0.236': high - 0.236 * diff,
                            'Level_0.382': high - 0.382 * diff,
                            'Level_0.5': high - 0.5 * diff,
                            'Level_0.618': high - 0.618 * diff,
                            'Level_1': low
                        }
                        return levels

                    recent_high = df['High'].max()
                    recent_low = df['Low'].min()
                    fib_levels = fibonacci_retracement_levels(recent_high, recent_low)

                    # Calculate Pivot Points
                    def pivot_points(df):
                        df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
                        df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
                        df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
                        df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
                        df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
                        return df

                    df = pivot_points(df)

                    # Function to generate buy/sell signals
                    def generate_signals(macd, signal, rsi, close):
                        buy_signals = [0] * len(macd)
                        sell_signals = [0] * len(macd)
                        for i in range(1, len(macd)):
                            if macd[i] > signal[i] and macd[i-1] <= signal[i-1]:
                                buy_signals[i] = 1
                            elif macd[i] < signal[i] and macd[i-1] >= signal[i-1]:
                                sell_signals[i] = 1
                        return buy_signals, sell_signals

                    df['Buy_Signal'], df['Sell_Signal'] = generate_signals(df['MACD'], df['MACD_Signal'], df['RSI'], df['Close'])

                    if analysis_type == "Trend Analysis":
                        st.subheader("Trend Analysis")

                        indicators = st.multiselect(
                            "Select Indicators",
                            ['Close', '20_MA', '50_MA', '200_MA', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI', 'Buy_Signal', 'Sell_Signal', 'ADX',
                            'Parabolic_SAR', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Middle', 'ATR'],
                            default=['Close', 'Buy_Signal', 'Sell_Signal']
                        )
                        timeframe = st.radio(
                            "Select Timeframe",
                            ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                            index=2,
                            horizontal=True
                        )

                        if timeframe == '15 days':
                            df = df[-15:]
                        elif timeframe == '30 days':
                            df = df[-30:]
                        elif timeframe == '90 days':
                            df = df[-90:]
                        elif timeframe == '180 days':
                            df = df[-180:]
                        elif timeframe == '1 year':
                            df = df[-365:]

                        fig = create_figure(df.set_index('Date'), indicators, f"Trend Analysis for {selected_stock}")

                        colors = {'Close': 'blue', '20_MA': 'orange', '50_MA': 'green', '200_MA': 'red', 'MACD': 'purple',
                                'MACD_Signal': 'brown', 'RSI': 'pink', 'Buy_Signal': 'green', 'Sell_Signal': 'red', 'ADX': 'magenta',
                                'Parabolic_SAR': 'yellow', 'Bollinger_High': 'black', 'Bollinger_Low': 'cyan',
                                'Bollinger_Middle': 'grey', 'ATR': 'darkblue'}

                        for indicator in indicators:
                            if indicator == 'Buy_Signal':
                                fig.add_trace(
                                    go.Scatter(x=df[df[indicator] == 1]['Date'],
                                            y=df[df[indicator] == 1]['Close'], mode='markers', name='Buy Signal',
                                            marker=dict(color='green', symbol='triangle-up')))
                            elif indicator == 'Sell_Signal':
                                fig.add_trace(
                                    go.Scatter(x=df[df[indicator] == 1]['Date'],
                                            y=df[df[indicator] == 1]['Close'], mode='markers', name='Sell Signal',
                                            marker=dict(color='red', symbol='triangle-down')))
                            elif indicator == 'MACD_Histogram':
                                fig.add_trace(
                                    go.Bar(x=df['Date'], y=df[indicator], name=indicator, marker_color='gray'))
                            else:
                                fig.add_trace(
                                    go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator,
                                            line=dict(color=colors.get(indicator, 'black'))))

                        st.plotly_chart(fig)

                    elif analysis_type == "Volume Analysis":
                        st.subheader("Volume Analysis")
                        volume_indicators = st.multiselect(
                            "Select Volume Indicators",
                            ['Close','Volume', 'Volume_MA_20', 'Volume_MA_10', 'Volume_MA_5', 'OBV', 'VWAP', 'A/D Line'],
                            default=['Close','VWAP']
                        )
                        volume_timeframe = st.radio(
                            "Select Timeframe",
                            ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                            index=2,
                            horizontal=True
                        )

                        if volume_timeframe == '15 days':
                            df = df[-15:]
                        elif volume_timeframe == '30 days':
                            df = df[-30:]
                        elif volume_timeframe == '90 days':
                            df = df[-90:]
                        elif volume_timeframe == '180 days':
                            df = df[-180:]
                        elif volume_timeframe == '1 year':
                            df = df[-365:]

                        fig = create_figure(df.set_index('Date'), volume_indicators, f"Volume Analysis for {selected_stock}")

                        for indicator in volume_indicators:
                            fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                        st.plotly_chart(fig)

                    elif analysis_type == "Support & Resistance Levels":
                        st.subheader("Support & Resistance Levels")
                        sr_indicators = st.multiselect(
                            "Select Indicators",
                            ['Close', '20_MA', '50_MA', '200_MA', 'Support', 'Resistance', 'Support_Trendline',
                            'Resistance_Trendline', 'Pivot', 'R1', 'S1', 'R2', 'S2'],
                            default=['Close', 'Support', 'Resistance']
                        )
                        sr_timeframe = st.radio(
                            "Select Timeframe",
                            ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                            index=2,
                            horizontal=True
                        )

                        if sr_timeframe == '15 days':
                            df = df[-15:]
                        elif sr_timeframe == '30 days':
                            df = df[-30:]
                        elif sr_timeframe == '90 days':
                            df = df[-90:]
                        elif sr_timeframe == '180 days':
                            df = df[-180:]
                        elif sr_timeframe == '1 year':
                            df = df[-365:]

                        fig = create_figure(df.set_index('Date'), sr_indicators, f"Support & Resistance Levels for {selected_stock}")

                        for indicator in sr_indicators:
                            fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                        st.plotly_chart(fig)


       
        elif choice == "Technical Analysis":
            #'Technical Analysis' code---------------------------------------------------------------------------------------------------------------------------------
            # Sidebar setup
            st.sidebar.subheader("Interactive Charts")

            # Sidebar for user input
            ticker = st.sidebar.text_input("Enter Stock Symbol", value='RVNL.NS')
            start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2023-01-01'))
            end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

            # Load stock data
            @st.cache_data
            def load_data(ticker, start, end):
                data = yf.download(ticker, start=start, end=end)
                data.reset_index(inplace=True)
                return data

            # Load index data
            @st.cache_data
            def load_index_data(ticker, start, end):
                data = yf.download(ticker, start=start, end=end)
                data.reset_index(inplace=True)
                return data

            st.title('Stock Technical Analysis')

            index_ticker = "^NSEI"  # NIFTY 50 index ticker

            # Load data
            data_load_state = st.text('Loading data...')
            data = load_data(ticker, start_date, end_date).copy()
            index_data = load_index_data(index_ticker, start_date, end_date).copy()
            data_load_state.text('Loading data...done!')

            # Calculate technical indicators
            def calculate_technical_indicators(df, index_df):
                # Moving averages
                df['10_SMA'] = ta.trend.sma_indicator(df['Close'], window=10)
                df['20_SMA'] = ta.trend.sma_indicator(df['Close'], window=20)
                df['10_EMA'] = ta.trend.ema_indicator(df['Close'], window=10)
                df['20_EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
                df['10_WMA'] = ta.trend.wma_indicator(df['Close'], window=10)
                df['20_WMA'] = ta.trend.wma_indicator(df['Close'], window=20)

                # Volume Moving Averages
                df['5_VMA'] = df['Volume'].rolling(window=5).mean()
                df['10_VMA'] = df['Volume'].rolling(window=10).mean()
                df['20_VMA'] = df['Volume'].rolling(window=20).mean()

                # Momentum Indicators
                df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                df['%K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
                df['%D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
                df['MACD'] = ta.trend.macd(df['Close'])
                df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

                # Volume Indicators
                df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
                df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
                df['A/D Line'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])

                # Volatility Indicators
                df['BB_High'], df['BB_Middle'], df['BB_Low'] = ta.volatility.bollinger_hband(df['Close']), ta.volatility.bollinger_mavg(df['Close']), ta.volatility.bollinger_lband(df['Close'])
                df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
                df['Std Dev'] = ta.volatility.bollinger_wband(df['Close'])

                # Trend Indicators
                df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
                df['+DI'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'])
                df['-DI'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'])
                psar = pta.psar(df['High'], df['Low'], df['Close'])
                df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']
                df['Ichimoku_a'] = ta.trend.ichimoku_a(df['High'], df['Low'])
                df['Ichimoku_b'] = ta.trend.ichimoku_b(df['High'], df['Low'])
                df['Ichimoku_base'] = ta.trend.ichimoku_base_line(df['High'], df['Low'])
                df['Ichimoku_conv'] = ta.trend.ichimoku_conversion_line(df['High'], df['Low'])

                # Support and Resistance Levels
                df = find_support_resistance(df)
                df['Support_Trendline'] = calculate_trendline(df, kind='support')
                df['Resistance_Trendline'] = calculate_trendline(df, kind='resistance')
                df = pivot_points(df)

                # Price Oscillators
                df['ROC'] = ta.momentum.roc(df['Close'], window=12)
                df['DPO'] = ta.trend.dpo(df['Close'], window=20)
                df['Williams %R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)

                # Market Breadth Indicators
                df['Advances'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else 0)
                df['Declines'] = df['Close'].diff().apply(lambda x: 1 if x < 0 else 0)
                df['McClellan Oscillator'] = (df['Advances'] - df['Declines']).rolling(window=19).mean() - (df['Advances'] - df['Declines']).rolling(window=39).mean()
                df['TRIN'] = (df['Advances'] / df['Declines']) / (df['Volume'][df['Advances'] > 0].sum() / df['Volume'][df['Declines'] > 0].sum())
                df['Advance-Decline Line'] = df['Advances'].cumsum() - df['Declines'].cumsum()

                # Relative Performance Indicators
                df['Price-to-Volume Ratio'] = df['Close'] / df['Volume']
                df['Relative Strength Comparison'] = df['Close'] / index_df['Close']
                df['Performance Relative to an Index'] = df['Close'].pct_change().cumsum() - index_df['Close'].pct_change().cumsum()

                return df

            # Identify Horizontal Support and Resistance
            def find_support_resistance(df, window=20):
                df['Support'] = df['Low'].rolling(window, center=True).min()
                df['Resistance'] = df['High'].rolling(window, center=True).max()
                return df

            # Draw Trendlines
            def calculate_trendline(df, kind='support'):
                if kind == 'support':
                    prices = df['Low']
                elif kind == 'resistance':
                    prices = df['High']
                else:
                    raise ValueError("kind must be either 'support' or 'resistance'")

                indices = np.arange(len(prices))
                slope, intercept, _, _, _ = linregress(indices, prices)
                trendline = slope * indices + intercept
                return trendline

            # Calculate Pivot Points
            def pivot_points(df):
                df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
                df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
                df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
                df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
                df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
                return df

            data = calculate_technical_indicators(data, index_data)

            # Function to add range buttons to the plot
            def add_range_buttons(fig):
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=14, label="14d", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True)
                    )
                )

            # Plotly visualization functions
            def plot_indicator(df, indicator, title, yaxis_title='Price', secondary_y=False):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator, yaxis="y2" if secondary_y else "y1"))
                
                if secondary_y:
                    fig.update_layout(
                        yaxis2=dict(
                            title=indicator,
                            overlaying='y',
                            side='right'
                        )
                    )
                
                fig.update_layout(title=title, xaxis_title='Date', yaxis_title=yaxis_title)
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_moving_average(df, ma_type):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='blue', opacity=0.5, yaxis='y2'))
                if ma_type == 'SMA':
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['10_SMA'], mode='lines', name='10_SMA'))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['20_SMA'], mode='lines', name='20_SMA'))
                elif ma_type == 'EMA':
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['10_EMA'], mode='lines', name='10_EMA'))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['20_EMA'], mode='lines', name='20_EMA'))
                elif ma_type == 'WMA':
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['10_WMA'], mode='lines', name='10_WMA'))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['20_WMA'], mode='lines', name='20_WMA'))
                fig.update_layout(title=f'{ma_type} Moving Averages', xaxis_title='Date', yaxis_title='Price', yaxis2=dict(title='Volume', overlaying='y', side='right'))
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_macd(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], mode='lines', name='MACD Signal'))

                # Plot MACD Histogram with different colors
                macd_hist_colors = []
                for i in range(1, len(df)):
                    if df['MACD_Hist'].iloc[i] > 0:
                        color = 'green' if df['MACD_Hist'].iloc[i] > df['MACD_Hist'].iloc[i - 1] else 'lightgreen'
                    else:
                        color = 'red' if df['MACD_Hist'].iloc[i] < df['MACD_Hist'].iloc[i - 1] else 'lightcoral'
                    macd_hist_colors.append(color)

                fig.add_trace(go.Bar(x=df['Date'][1:], y=df['MACD_Hist'][1:], name='MACD Histogram', marker_color=macd_hist_colors, yaxis='y2'))

                fig.update_layout(
                    title='MACD',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    yaxis2=dict(
                        title='MACD Histogram',
                        overlaying='y',
                        side='right'
                    )
                )
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_trendlines(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                fig.add_trace(go.Scatter(x=df['Date'], y=df['Support_Trendline'], mode='lines', name='Support Trendline', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Resistance_Trendline'], mode='lines', name='Resistance Trendline', line=dict(color='red', dash='dash')))

                fig.update_layout(title='Trendlines', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_fibonacci_retracement(df):
                high = df['High'].max()
                low = df['Low'].min()

                diff = high - low
                levels = [high, high - 0.236 * diff, high - 0.382 * diff, high - 0.5 * diff, high - 0.618 * diff, low]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                for level in levels:
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                            y=[level, level],
                                            mode='lines', name=f'Level {level}', line=dict(dash='dash')))

                fig.update_layout(title='Fibonacci Retracement Levels', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_gann_fan_lines(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                # Adding Gann fan lines (simple example, for more advanced lines use a proper method)
                for i in range(1, 5):
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                            y=[df['Close'].iloc[0], df['Close'].iloc[0] + i * (df['Close'].iloc[-1] - df['Close'].iloc[0]) / 4],
                                            mode='lines', name=f'Gann Fan {i}', line=dict(dash='dash')))

                fig.update_layout(title='Gann Fan Lines', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_chart_patterns(df, pattern):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                # Adding example chart patterns (simple example, for more advanced patterns use a proper method)
                pattern_data = detect_chart_patterns(df, pattern)
                if pattern_data:
                    for pattern_info in pattern_data:
                        fig.add_trace(go.Scatter(x=pattern_info['x'], y=pattern_info['y'], mode='lines+markers', name=pattern_info['name'], line=dict(color=pattern_info['color'])))

                fig.update_layout(title=f'{pattern}', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

            def detect_chart_patterns(df, pattern):
                patterns = []
                if pattern == 'Head and Shoulders':
                    patterns = detect_head_and_shoulders(df)
                elif pattern == 'Double Tops and Bottoms':
                    patterns = detect_double_tops_and_bottoms(df)
                elif pattern == 'Flags and Pennants':
                    patterns = detect_flags_and_pennants(df)
                elif pattern == 'Triangles':
                    patterns = detect_triangles(df)
                elif pattern == 'Cup and Handle':
                    patterns = detect_cup_and_handle(df)
                return patterns

            def detect_head_and_shoulders(df):
                patterns = []
                window = 20  # Sliding window size
                for i in range(window, len(df) - window):
                    window_df = df.iloc[i - window:i + window]
                    if len(window_df) < 3:  # Ensure there are enough data points
                        continue
                    max_high = window_df['High'].max()
                    min_low = window_df['Low'].min()
                    middle_idx = window_df['High'].idxmax()
                    left_idx = window_df.iloc[:middle_idx]['High'].idxmax()
                    right_idx = window_df.iloc[middle_idx + 1:]['High'].idxmax()

                    if window_df['High'].iloc[left_idx] < max_high and window_df['High'].iloc[right_idx] < max_high and \
                            window_df['Low'].iloc[middle_idx] > min_low:
                        patterns.append({
                            "x": [window_df['Date'].iloc[left_idx], window_df['Date'].iloc[middle_idx], window_df['Date'].iloc[right_idx]],
                            "y": [window_df['High'].iloc[left_idx], window_df['High'].iloc[middle_idx], window_df['High'].iloc[right_idx]],
                            "name": "Head and Shoulders",
                            "color": "orange"
                        })
                return patterns

            def detect_double_tops_and_bottoms(df):
                patterns = []
                window = 20  # Sliding window size
                for i in range(window, len(df) - window):
                    window_df = df.iloc[i - window:i + window]
                    if len(window_df) < 3:  # Ensure there are enough data points
                        continue
                    max_high = window_df['High'].max()
                    min_low = window_df['Low'].min()
                    double_top = window_df['High'].value_counts().get(max_high, 0) > 1
                    double_bottom = window_df['Low'].value_counts().get(min_low, 0) > 1

                    if double_top:
                        patterns.append({
                            "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                            "y": [max_high, max_high],
                            "name": "Double Top",
                            "color": "red"
                        })
                    elif double_bottom:
                        patterns.append({
                            "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                            "y": [min_low, min_low],
                            "name": "Double Bottom",
                            "color": "green"
                        })
                return patterns

            def detect_flags_and_pennants(df):
                patterns = []
                window = 20  # Sliding window size
                for i in range(window, len(df) - window):
                    window_df = df.iloc[i - window:i + window]
                    if len(window_df) < 3:  # Ensure there are enough data points
                        continue
                    min_low = window_df['Low'].min()
                    max_high = window_df['High'].max()
                    flag_pattern = ((window_df['High'] - window_df['Low']) / window_df['Low']).mean() < 0.05

                    if flag_pattern:
                        patterns.append({
                            "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                            "y": [min_low, max_high],
                            "name": "Flag",
                            "color": "purple"
                        })
                return patterns

            def detect_triangles(df):
                patterns = []
                window = 20  # Sliding window size
                for i in range(window, len(df) - window):
                    window_df = df.iloc[i - window:i + window]
                    if len(window_df) < 3:  # Ensure there are enough data points
                        continue
                    max_high = window_df['High'].max()
                    min_low = window_df['Low'].min()
                    triangle_pattern = np.all(np.diff(window_df['High']) < 0) and np.all(np.diff(window_df['Low']) > 0)

                    if triangle_pattern:
                        patterns.append({
                            "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                            "y": [min_low, max_high],
                            "name": "Triangle",
                            "color": "blue"
                        })
                return patterns

            def detect_cup_and_handle(df):
                patterns = []
                window = 50  # Sliding window size
                for i in range(window, len(df) - window):
                    window_df = df.iloc[i - window:i + window]
                    if len(window_df) < 3:  # Ensure there are enough data points
                        continue
                    max_high = window_df['High'].max()
                    min_low = window_df['Low'].min()
                    cup_shape = ((window_df['High'] - window_df['Low']) / window_df['Low']).mean() < 0.1

                    if cup_shape:
                        patterns.append({
                            "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[len(window_df) // 2], window_df['Date'].iloc[-1]],
                            "y": [max_high, min_low, max_high],
                            "name": "Cup and Handle",
                            "color": "brown"
                        })
                return patterns

            def plot_mcclellan_oscillator(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['McClellan Oscillator'], mode='lines', name='McClellan Oscillator'))
                fig.update_layout(title='McClellan Oscillator', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_trin(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['TRIN'], mode='lines', name='TRIN'))
                fig.update_layout(title='Arms Index (TRIN)', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def get_signals(df):
                signals = []

                # Example logic for signals (these can be customized)
                if df['Close'].iloc[-1] > df['20_SMA'].iloc[-1]:
                    signals.append(("Simple Moving Average (20_SMA)", "Hold", "Price is above the SMA."))
                else:
                    signals.append(("Simple Moving Average (20_SMA)", "Sell", "Price crossed below the SMA."))

                if df['Close'].iloc[-1] > df['20_EMA'].iloc[-1]:
                    signals.append(("Exponential Moving Average (20_EMA)", "Hold", "Price is above the EMA."))
                else:
                    signals.append(("Exponential Moving Average (20_EMA)", "Sell", "Price crossed below the EMA."))

                if df['Close'].iloc[-1] > df['20_WMA'].iloc[-1]:
                    signals.append(("Weighted Moving Average (20_WMA)", "Hold", "Price is above the WMA."))
                else:
                    signals.append(("Weighted Moving Average (20_WMA)", "Sell", "Price crossed below the WMA."))

                if df['RSI'].iloc[-1] < 30:
                    signals.append(("Relative Strength Index (RSI)", "Buy", "RSI crosses below 30 (oversold)."))
                elif df['RSI'].iloc[-1] > 70:
                    signals.append(("Relative Strength Index (RSI)", "Sell", "RSI crosses above 70 (overbought)."))
                else:
                    signals.append(("Relative Strength Index (RSI)", "Hold", "RSI is between 30 and 70."))

                if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                    signals.append(("Moving Average Convergence Divergence (MACD)", "Buy", "MACD line crosses above the signal line."))
                else:
                    signals.append(("Moving Average Convergence Divergence (MACD)", "Sell", "MACD line crosses below the signal line."))

                if df['%K'].iloc[-1] < 20 and df['%D'].iloc[-1] < 20 and df['%K'].iloc[-1] > df['%D'].iloc[-1]:
                    signals.append(("Stochastic Oscillator", "Buy", "%K line crosses above %D line and both are below 20."))
                elif df['%K'].iloc[-1] > 80 and df['%D'].iloc[-1] > 80 and df['%K'].iloc[-1] < df['%D'].iloc[-1]:
                    signals.append(("Stochastic Oscillator", "Sell", "%K line crosses below %D line and both are above 80."))
                else:
                    signals.append(("Stochastic Oscillator", "Hold", "No clear buy or sell signal."))

                if df['OBV'].diff().iloc[-1] > 0:
                    signals.append(("On-Balance Volume (OBV)", "Buy", "OBV is increasing."))
                else:
                    signals.append(("On-Balance Volume (OBV)", "Sell", "OBV is decreasing."))

                if df['Close'].iloc[-1] > df['VWAP'].iloc[-1]:
                    signals.append(("Volume Weighted Average Price (VWAP)", "Buy", "Price crosses above the VWAP."))
                else:
                    signals.append(("Volume Weighted Average Price (VWAP)", "Sell", "Price crosses below the VWAP."))

                if df['A/D Line'].diff().iloc[-1] > 0:
                    signals.append(("Accumulation/Distribution Line (A/D Line)", "Buy", "A/D Line is increasing."))
                else:
                    signals.append(("Accumulation/Distribution Line (A/D Line)", "Sell", "A/D Line is decreasing."))

                if df['Close'].iloc[-1] < df['BB_Low'].iloc[-1]:
                    signals.append(("Bollinger Bands", "Buy", "Price crosses below the lower band."))
                elif df['Close'].iloc[-1] > df['BB_High'].iloc[-1]:
                    signals.append(("Bollinger Bands", "Sell", "Price crosses above the upper band."))
                else:
                    signals.append(("Bollinger Bands", "Hold", "Price is within Bollinger Bands."))

                if df['ATR'].iloc[-1] > df['ATR'].rolling(window=14).mean().iloc[-1]:
                    signals.append(("Average True Range (ATR)", "Buy", "ATR is increasing, indicating higher volatility."))
                else:
                    signals.append(("Average True Range (ATR)", "Sell", "ATR is decreasing, indicating lower volatility."))

                if df['Std Dev'].iloc[-1] > df['Close'].rolling(window=20).std().iloc[-1]:
                    signals.append(("Standard Deviation", "Buy", "Price is below the mean minus 2 standard deviations."))
                else:
                    signals.append(("Standard Deviation", "Sell", "Price is above the mean plus 2 standard deviations."))

                if df['Parabolic_SAR'].iloc[-1] < df['Close'].iloc[-1]:
                    signals.append(("Parabolic SAR (Stop and Reverse)", "Buy", "Price crosses above the SAR."))
                else:
                    signals.append(("Parabolic SAR (Stop and Reverse)", "Sell", "Price crosses below the SAR."))

                if df['ROC'].iloc[-1] > 0:
                    signals.append(("Price Rate of Change (ROC)", "Buy", "ROC crosses above zero."))
                else:
                    signals.append(("Price Rate of Change (ROC)", "Sell", "ROC crosses below zero."))

                if df['DPO'].iloc[-1] > 0:
                    signals.append(("Detrended Price Oscillator (DPO)", "Buy", "DPO crosses above zero."))
                else:
                    signals.append(("Detrended Price Oscillator (DPO)", "Sell", "DPO crosses below zero."))

                if df['Williams %R'].iloc[-1] < -80:
                    signals.append(("Williams %R", "Buy", "Williams %R crosses above -80 (indicating oversold)."))
                elif df['Williams %R'].iloc[-1] > -20:
                    signals.append(("Williams %R", "Sell", "Williams %R crosses below -20 (indicating overbought)."))
                else:
                    signals.append(("Williams %R", "Hold", "Williams %R is between -80 and -20."))

                if df['Close'].iloc[-1] > df['Pivot'].iloc[-1]:
                    signals.append(("Pivot Points", "Buy", "Price crosses above the pivot point."))
                else:
                    signals.append(("Pivot Points", "Sell", "Price crosses below the pivot point."))

                high = df['High'].max()
                low = df['Low'].min()
                diff = high - low
                fib_levels = [high, high - 0.236 * diff, high - 0.382 * diff, high - 0.5 * diff, high - 0.618 * diff, low]
                for level in fib_levels:
                    if df['Close'].iloc[-1] > level:
                        signals.append(("Fibonacci Retracement Levels", "Buy", "Price crosses above a Fibonacci retracement level."))
                        break
                    elif df['Close'].iloc[-1] < level:
                        signals.append(("Fibonacci Retracement Levels", "Sell", "Price crosses below a Fibonacci retracement level."))
                        break

                gann_fan_line = [df['Close'].iloc[0] + i * (df['Close'].iloc[-1] - df['Close'].iloc[0]) / 4 for i in range(1, 5)]
                for line in gann_fan_line:
                    if df['Close'].iloc[-1] > line:
                        signals.append(("Gann Fan Lines", "Buy", "Price crosses above a Gann fan line."))
                        break
                    elif df['Close'].iloc[-1] < line:
                        signals.append(("Gann Fan Lines", "Sell", "Price crosses below a Gann fan line."))
                        break

                if df['McClellan Oscillator'].iloc[-1] > 0:
                    signals.append(("McClellan Oscillator", "Buy", "Oscillator crosses above zero."))
                else:
                    signals.append(("McClellan Oscillator", "Sell", "Oscillator crosses below zero."))

                if df['TRIN'].iloc[-1] < 1:
                    signals.append(("Arms Index (TRIN)", "Buy", "TRIN below 1.0 (more advancing volume)."))
                else:
                    signals.append(("Arms Index (TRIN)", "Sell", "TRIN above 1.0 (more declining volume)."))

                # Chart Patterns
                patterns = detect_chart_patterns(df, 'Summary')
                signals.extend(patterns)

                # Additional Indicators
                if df['Ichimoku_a'].iloc[-1] > df['Ichimoku_b'].iloc[-1]:
                    signals.append(("Ichimoku Cloud", "Buy", "Ichimoku conversion line above baseline."))
                else:
                    signals.append(("Ichimoku Cloud", "Sell", "Ichimoku conversion line below baseline."))

                if df['Relative Strength Comparison'].iloc[-1] > 1:
                    signals.append(("Relative Strength Comparison", "Buy", "Stock outperforms index."))
                else:
                    signals.append(("Relative Strength Comparison", "Sell", "Stock underperforms index."))

                if df['Performance Relative to an Index'].iloc[-1] > 0:
                    signals.append(("Performance Relative to an Index", "Buy", "Stock outperforms index over time."))
                else:
                    signals.append(("Performance Relative to an Index", "Sell", "Stock underperforms index over time."))

                if df['Advance-Decline Line'].diff().iloc[-1] > 0:
                    signals.append(("Advance-Decline Line", "Buy", "Advances exceed declines."))
                else:
                    signals.append(("Advance-Decline Line", "Sell", "Declines exceed advances."))

                if df['Price-to-Volume Ratio'].iloc[-1] > df['Price-to-Volume Ratio'].rolling(window=14).mean().iloc[-1]:
                    signals.append(("Price-to-Volume Ratio", "Buy", "Price-to-Volume ratio increasing."))
                else:
                    signals.append(("Price-to-Volume Ratio", "Sell", "Price-to-Volume ratio decreasing."))

                return signals

            signals = get_signals(data)

            # Sidebar for technical indicators
            st.sidebar.header('Technical Indicators')
            indicator_category = st.sidebar.radio('Select Indicator Category', [
                'Moving Averages', 'Momentum Indicators', 'Volume Indicators', 'Volatility Indicators', 'Trend Indicators',
                'Support and Resistance Levels', 'Price Oscillators', 'Market Breadth Indicators', 'Chart Patterns', 'Relative Performance Indicators', 'Summary'
            ])

            # Display technical indicators
            st.subheader('Technical Indicators')
            if indicator_category != 'Summary':
                if indicator_category == 'Moving Averages':
                    indicators = st.selectbox("Select Moving Average", ['SMA', 'EMA', 'WMA'])
                    plot_moving_average(data, indicators)
                elif indicator_category == 'Momentum Indicators':
                    indicators = st.selectbox("Select Momentum Indicator", ['RSI', 'Stochastic Oscillator', 'MACD'])
                    if indicators == 'RSI':
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=[70] * len(data), mode='lines', name='RSI 70', line=dict(color='red', dash='dash')))
                        fig.add_trace(go.Scatter(x=data['Date'], y=[30] * len(data), mode='lines', name='RSI 30', line=dict(color='green', dash='dash')))
                        fig.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='Value')
                        add_range_buttons(fig)
                        st.plotly_chart(fig)
                    elif indicators == 'Stochastic Oscillator':
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['%K'], mode='lines', name='%K'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['%D'], mode='lines', name='%D'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=[80] * len(data), mode='lines', name='%K 80', line=dict(color='red', dash='dash')))
                        fig.add_trace(go.Scatter(x=data['Date'], y=[20] * len(data), mode='lines', name='%K 20', line=dict(color='green', dash='dash')))
                        fig.update_layout(title='Stochastic Oscillator', xaxis_title='Date', yaxis_title='Value')
                        add_range_buttons(fig)
                        st.plotly_chart(fig)
                    elif indicators == 'MACD':
                        plot_macd(data)
                elif indicator_category == 'Volume Indicators':
                    indicators = st.selectbox("Select Volume Indicator", ['OBV', 'VWAP', 'A/D Line', 'Volume Moving Averages'])
                    if indicators == 'OBV':
                        plot_indicator(data, 'OBV', 'On-Balance Volume (OBV)')
                    elif indicators == 'VWAP':
                        plot_indicator(data, 'VWAP', 'Volume Weighted Average Price (VWAP)')
                    elif indicators == 'A/D Line':
                        plot_indicator(data, 'A/D Line', 'Accumulation/Distribution Line')
                    elif indicators == 'Volume Moving Averages':
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume', marker_color='blue', opacity=0.5))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['5_VMA'], mode='lines', name='5_VMA'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['10_VMA'], mode='lines', name='10_VMA'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['20_VMA'], mode='lines', name='20_VMA'))
                        fig.update_layout(title='Volume Moving Averages', xaxis_title='Date', yaxis_title='Volume')
                        add_range_buttons(fig)
                        st.plotly_chart(fig)
                elif indicator_category == 'Volatility Indicators':
                    indicators = st.selectbox("Select Volatility Indicator", ['Bollinger Bands', 'ATR', 'Standard Deviation'])
                    if indicators == 'Bollinger Bands':
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_High'], mode='lines', name='BB High'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Middle'], mode='lines', name='BB Middle'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Low'], mode='lines', name='BB Low'))
                        fig.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
                        add_range_buttons(fig)
                        st.plotly_chart(fig)
                    elif indicators == 'ATR':
                        plot_indicator(data, 'ATR', 'Average True Range (ATR)')
                    elif indicators == 'Standard Deviation':
                        plot_indicator(data, 'Std Dev', 'Standard Deviation')
                elif indicator_category == 'Trend Indicators':
                    indicators = st.selectbox("Select Trend Indicator", ['Trendlines', 'Parabolic SAR', 'Ichimoku Cloud', 'ADX'])
                    if indicators == 'Trendlines':
                        plot_trendlines(data)
                    elif indicators == 'Parabolic SAR':
                        plot_indicator(data, 'Parabolic_SAR', 'Parabolic SAR')
                    elif indicators == 'Ichimoku Cloud':
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_a'], mode='lines', name='Ichimoku A'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_b'], mode='lines', name='Ichimoku B'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_base'], mode='lines', name='Ichimoku Base Line'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_conv'], mode='lines', name='Ichimoku Conversion Line'))
                        fig.update_layout(title='Ichimoku Cloud', xaxis_title='Date', yaxis_title='Value')
                        add_range_buttons(fig)
                        st.plotly_chart(fig)
                    elif indicators == 'ADX':
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['ADX'], mode='lines', name='ADX'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['+DI'], mode='lines', name='+DI'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['-DI'], mode='lines', name='-DI'))
                        fig.update_layout(title='Average Directional Index (ADX)', xaxis_title='Date', yaxis_title='Value')
                        add_range_buttons(fig)
                        st.plotly_chart(fig)
                elif indicator_category == 'Support and Resistance Levels':
                    indicators = st.selectbox("Select Support and Resistance Level", ['Pivot Points', 'Fibonacci Retracement Levels', 'Gann Fan Lines'])
                    if indicators == 'Pivot Points':
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Pivot'], mode='lines', name='Pivot'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['R1'], mode='lines', name='R1'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['S1'], mode='lines', name='S1'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['R2'], mode='lines', name='R2'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['S2'], mode='lines', name='S2'))
                        fig.update_layout(title='Pivot Points', xaxis_title='Date', yaxis_title='Price')
                        add_range_buttons(fig)
                        st.plotly_chart(fig)
                    elif indicators == 'Fibonacci Retracement Levels':
                        plot_fibonacci_retracement(data)
                    elif indicators == 'Gann Fan Lines':
                        plot_gann_fan_lines(data)
                elif indicator_category == 'Price Oscillators':
                    indicators = st.selectbox("Select Price Oscillator", ['ROC', 'DPO', 'Williams %R'])
                    if indicators == 'ROC':
                        plot_indicator(data, 'ROC', 'Rate of Change (ROC)')
                    elif indicators == 'DPO':
                        plot_indicator(data, 'DPO', 'Detrended Price Oscillator (DPO)')
                    elif indicators == 'Williams %R':
                        plot_indicator(data, 'Williams %R', 'Williams %R')
                elif indicator_category == 'Market Breadth Indicators':
                    indicators = st.selectbox("Select Market Breadth Indicator", ['Advance-Decline Line', 'McClellan Oscillator', 'TRIN'])
                    if indicators == 'Advance-Decline Line':
                        plot_indicator(data, 'Advance-Decline Line', 'Advance-Decline Line')
                    elif indicators == 'McClellan Oscillator':
                        plot_mcclellan_oscillator(data)
                    elif indicators == 'TRIN':
                        plot_trin(data)
                elif indicator_category == 'Chart Patterns':
                    indicators = st.selectbox("Select Chart Pattern", ['Head and Shoulders', 'Double Tops and Bottoms', 'Flags and Pennants', 'Triangles', 'Cup and Handle'])
                    if indicators == 'Head and Shoulders':
                        plot_chart_patterns(data, 'Head and Shoulders')
                    elif indicators == 'Double Tops and Bottoms':
                        plot_chart_patterns(data, 'Double Tops and Bottoms')
                    elif indicators == 'Flags and Pennants':
                        plot_chart_patterns(data, 'Flags and Pennants')
                    elif indicators == 'Triangles':
                        plot_chart_patterns(data, 'Triangles')
                    elif indicators == 'Cup and Handle':
                        plot_chart_patterns(data, 'Cup and Handle')
                elif indicator_category == 'Relative Performance Indicators':
                    indicators = st.selectbox("Select Relative Performance Indicator", ['Price-to-Volume Ratio', 'Relative Strength Comparison', 'Performance Relative to an Index'])
                    if indicators == 'Price-to-Volume Ratio':
                        plot_indicator(data, 'Price-to-Volume Ratio', 'Price-to-Volume Ratio', secondary_y=True)
                    elif indicators == 'Relative Strength Comparison':
                        plot_indicator(data, 'Relative Strength Comparison', 'Relative Strength Comparison')
                    elif indicators == 'Performance Relative to an Index':
                        plot_indicator(data, 'Performance Relative to an Index', 'Performance Relative to an Index')
            else:
                # Display signals in a dataframe with improved visualization
                st.subheader('Technical Indicator Signals')
                signals_df = pd.DataFrame(signals, columns=['Technical Indicator', 'Signal', 'Reason'])
                st.dataframe(signals_df.style.applymap(lambda x: 'background-color: lightgreen' if 'Buy' in x else 'background-color: lightcoral' if 'Sell' in x else '', subset=['Signal']))

         
        elif choice == "Stock Prediction":
            # Your existing 'Stock Price Forecasting' code-----------------------------------------------------------------------------------------------------------------------


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
                    (results_df['Wavelet_Max_Amplitude'] > 1000) &  # Adjust threshold as necessary
                    (results_df['Hilbert_Amplitude'] > results_df['Hilbert_Amplitude'].mean()) &  # Strong current price movement
                    (results_df['Fourier_Trend'] == 'upward')  # Current price trend is upward
                ]

                if not buy_recommendation.empty:
                    st.write("**Recommendation: Buy**")
                    st.dataframe(buy_recommendation[['Ticker', 'Fourier_Dominant_Cycle_Days', 'Fourier_Angle', 'Fourier_Trend', 'Wavelet_Max_Amplitude', 'Hilbert_Amplitude']])
                else:
                    st.write("No strong buy recommendations based on the current analysis.")

            else:
                pass


        elif choice == "Stock Watch":
            # Your existing 'Stock Watch' code----------------------------------------------------------------------------------------------------------------------------------
            st.sidebar.subheader("Strategies")

            submenu = st.sidebar.selectbox("Select Strategy", ["MACD", "Moving Average", "RSI", "Bollinger Bands", "Stochastic Oscillator",
                "Ichimoku Cloud", "ADX", "Fibonacci Retracement", "Parabolic SAR",
                "Candlestick Patterns", "Pivot Points"])

            # Dropdown for selecting ticker category
            ticker_category = st.sidebar.selectbox("Select Ticker Category", ["MAD-LargeCap", "MAD-MidCap","BSE-LargeCap","BSE-MidCap","BSE-SmallCap"])

            # Set tickers based on selected category
            if ticker_category == "MAD-LargeCap":
                tickers = largecap_tickers
            elif ticker_category == "MAD-MidCap":
                tickers = midcap_tickers
            elif ticker_category == "BSE-LargeCap":
                tickers = bse_largecap
            elif ticker_category == "BSE-MidCap":
                tickers = bse_midcap
            else:
                tickers = bse_smallcap

            # Function to calculate MACD and related values
            def calculate_macd(data, slow=26, fast=12, signal=9):
                data['EMA_fast'] = data['Close'].ewm(span=fast, min_periods=fast).mean()
                data['EMA_slow'] = data['Close'].ewm(span=slow, min_periods=slow).mean()
                data['MACD'] = data['EMA_fast'] - data['EMA_slow']
                data['MACD_signal'] = data['MACD'].ewm(span=signal, min_periods=signal).mean()
                data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
                return data

            # Function to calculate RSI
            def calculate_rsi(data, period=14):
                rsi_indicator = ta.momentum.RSIIndicator(close=data['Close'], window=period)
                data['RSI'] = rsi_indicator.rsi()
                return data

            # Function to calculate ADX and related values
            def calculate_adx(data, period=14):
                adx_indicator = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=period)
                data['ADX'] = adx_indicator.adx()
                data['+DI'] = adx_indicator.adx_pos()
                data['-DI'] = adx_indicator.adx_neg()
                return data

            # Function to check MACD < MACD signal
            def check_macd_signal(data):
                return data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]

            # Function to check the second criteria
            def check_negative_histogram_and_price(data):
                histogram_increasing = (data['MACD_histogram'].iloc[-3] <= data['MACD_histogram'].iloc[-2] <= data['MACD_histogram'].iloc[-1])
                histogram_negative = data['MACD_histogram'].iloc[-1] < 0
                price_increasing = (data['Close'].iloc[-1] >= data['Close'].iloc[-2] >= data['Close'].iloc[-3] >= data['Close'].iloc[-4])
                return histogram_increasing, histogram_negative, price_increasing

            # Function to fetch and process stock data
            @st.cache_data(ttl=3600)
            def get_stock_data(ticker_symbols, period, interval):
                try:
                    stock_data = {}
                    progress_bar = st.progress(0)
                    for idx, ticker_symbol in enumerate(ticker_symbols):
                        df = yf.download(ticker_symbol, period=period, interval=interval)
                        if not df.empty:
                            df.interpolate(method='linear', inplace=True)
                            df = calculate_indicators(df)
                            df.dropna(inplace=True)
                            stock_data[ticker_symbol] = df
                        progress_bar.progress((idx + 1) / len(ticker_symbols))
                    return stock_data
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
                    return {}

            # Function to calculate essential technical indicators
            @st.cache_data(ttl=3600)
            def calculate_indicators(df):
                # Calculate Moving Averages
                df['20_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=20).wma()
                df['50_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=50).wma()

                # Calculate MACD
                macd = ta.trend.MACD(df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_Signal'] = macd.macd_signal()
                df['MACD_Histogram'] = macd.macd_diff()

                # Calculate RSI
                rsi = ta.momentum.RSIIndicator(df['Close'])
                df['RSI'] = rsi.rsi()

                # Calculate Bollinger Bands
                bollinger = ta.volatility.BollingerBands(df['Close'])
                df['Bollinger_High'] = bollinger.bollinger_hband()
                df['Bollinger_Low'] = bollinger.bollinger_lband()
                df['Bollinger_Middle'] = bollinger.bollinger_mavg()

                # Calculate Stochastic Oscillator
                stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
                df['%K'] = stoch.stoch()
                df['%D'] = stoch.stoch_signal()

                # Calculate Ichimoku Cloud
                ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
                df['Tenkan_Sen'] = ichimoku.ichimoku_conversion_line()
                df['Kijun_Sen'] = ichimoku.ichimoku_base_line()
                df['Senkou_Span_A'] = ichimoku.ichimoku_a()
                df['Senkou_Span_B'] = ichimoku.ichimoku_b()
                # Chikou Span (Lagging Span) is the close price shifted by 26 periods
                df['Chikou_Span'] = df['Close'].shift(-26)

                # Calculate Parabolic SAR
                psar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'])
                df['Parabolic_SAR'] = psar.psar()

                return df

            # Calculate exponential moving averages
            def calculate_ema(data, short_window, long_window):
                data['Short_EMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
                data['Long_EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
                return data

            # Check if 50-day EMA crossed above 200-day EMA in the last 5 days
            def check_moving_average_crossover(data):
                recent_data = data[-5:]
                for i in range(1, len(recent_data)):
                    if (recent_data['Short_EMA'].iloc[i] > recent_data['Long_EMA'].iloc[i] and
                        recent_data['Short_EMA'].iloc[i-1] <= recent_data['Long_EMA'].iloc[i-1]):
                        return True
                return False

            # Check if price crossed below Bollinger Low in the last 5 days
            def check_bollinger_low_cross(data):
                recent_data = data[-5:]
                for i in range(1, len(recent_data)):
                    if (recent_data['Close'].iloc[i] < recent_data['Bollinger_Low'].iloc[i] and
                        recent_data['Close'].iloc[i-1] >= recent_data['Bollinger_Low'].iloc[i-1]):
                        return True
                return False

            # Check if RSI is below 30
            def check_rsi(data):
                return data['RSI'].iloc[-1] < 30

            # Check if Stochastic %K crossed above %D from below 20 in the last 5 days
            def check_stochastic(data):
                recent_data = data[-5:]
                for i in range(1, len(recent_data)):
                    if (recent_data['%K'].iloc[i] > recent_data['%D'].iloc[i] and
                        recent_data['%K'].iloc[i-1] <= recent_data['%D'].iloc[i-1] and
                        recent_data['%K'].iloc[i] < 20):
                        return True
                return False

            # Check if price is above Ichimoku Cloud
            def check_ichimoku(data):
                return data['Close'].iloc[-1] > data['Senkou_Span_A'].iloc[-1] and data['Close'].iloc[-1] > data['Senkou_Span_B'].iloc[-1]

            # Check if +DI crossed above -DI in the last 5 days
            def check_adx(data):
                recent_data = data[-5:]
                for i in range(1, len(recent_data)):
                    if (recent_data['+DI'].iloc[i] > recent_data['-DI'].iloc[i] and
                        recent_data['+DI'].iloc[i-1] <= recent_data['-DI'].iloc[i-1]):
                        return True
                return False

            # Check if price crossed above Parabolic SAR
            def check_parabolic_sar(data):
                recent_data = data[-5:]
                for i in range(1, len(recent_data)):
                    if (recent_data['Close'].iloc[i] > recent_data['Parabolic_SAR'].iloc[i] and
                        recent_data['Close'].iloc[i-1] <= recent_data['Parabolic_SAR'].iloc[i-1]):
                        return True
                return False

            # Calculate pivot points and their respective support and resistance levels
            def calculate_pivot_points(data, period='daily'):
                if period == 'daily':
                    data['Pivot'] = (data['High'].shift(1) + data['Low'].shift(1) + data['Close'].shift(1)) / 3
                elif period == 'weekly':
                    data['Pivot'] = (data['High'].resample('W').max().shift(1) + data['Low'].resample('W').min().shift(1) + data['Close'].resample('W').last().shift(1)) / 3
                elif period == 'monthly':
                    data['Pivot'] = (data['High'].resample('M').max().shift(1) + data['Low'].resample('M').min().shift(1) + data['Close'].resample('M').last().shift(1)) / 3

                data['R1'] = 2 * data['Pivot'] - data['Low']
                data['S1'] = 2 * data['Pivot'] - data['High']
                data['R2'] = data['Pivot'] + (data['High'] - data['Low'])
                data['S2'] = data['Pivot'] - (data['High'] - data['Low'])
                data['R3'] = data['Pivot'] + 2 * (data['High'] - data['Low'])
                data['S3'] = data['Pivot'] - 2 * (data['High'] - data['Low'])
                
                return data

            # Check for trading breakouts above resistance levels
            def check_pivot_breakouts(data):
                recent_data = data[-5:]
                for i in range(1, len(recent_data)):
                    if (recent_data['Close'].iloc[i] > recent_data['R1'].iloc[i] and recent_data['Close'].iloc[i-1] <= recent_data['R1'].iloc[i-1]) or \
                    (recent_data['Close'].iloc[i] > recent_data['R2'].iloc[i] and recent_data['Close'].iloc[i-1] <= recent_data['R2'].iloc[i-1]) or \
                    (recent_data['Close'].iloc[i] > recent_data['R3'].iloc[i] and recent_data['Close'].iloc[i-1] <= recent_data['R3'].iloc[i-1]):
                        return True
                return False

            macd_signal_list = []
            negative_histogram_tickers = []
            moving_average_tickers = []
            bollinger_low_cross_tickers = []
            rsi_tickers = []
            stochastic_tickers = []
            ichimoku_tickers = []
            adx_tickers = []
            parabolic_sar_tickers = []
            pivot_point_tickers = []

            progress_bar = st.progress(0)
            progress_step = 1 / len(tickers)

            for i, ticker in enumerate(tickers):
                progress_bar.progress((i + 1) * progress_step)
                data = yf.download(ticker, period="1y", interval="1d")
                if data.empty:
                    continue
                data = calculate_indicators(data)
                if submenu == "MACD":
                    data = calculate_macd(data)
                    data = calculate_rsi(data)
                    data = calculate_adx(data)
                    if check_macd_signal(data):
                        macd_signal_list.append(ticker)
                    histogram_increasing, histogram_negative, price_increasing = check_negative_histogram_and_price(data)
                    if histogram_increasing and histogram_negative and price_increasing:
                        negative_histogram_tickers.append(ticker)
                elif submenu == "Moving Average":
                    data = calculate_ema(data, short_window=50, long_window=200)
                    if check_moving_average_crossover(data):
                        moving_average_tickers.append(ticker)
                elif submenu == "Bollinger Bands":
                    if check_bollinger_low_cross(data):
                        bollinger_low_cross_tickers.append(ticker)
                elif submenu == "RSI":
                    if check_rsi(data):
                        rsi_tickers.append(ticker)
                elif submenu == "Stochastic Oscillator":
                    if check_stochastic(data):
                        stochastic_tickers.append(ticker)
                elif submenu == "Ichimoku Cloud":
                    if check_ichimoku(data):
                        ichimoku_tickers.append(ticker)
                elif submenu == "ADX":
                    data = calculate_adx(data)
                    if check_adx(data):
                        adx_tickers.append(ticker)
                elif submenu == "Parabolic SAR":
                    if check_parabolic_sar(data):
                        parabolic_sar_tickers.append(ticker)
                elif submenu == "Pivot Points":
                    data = calculate_pivot_points(data, period='daily')
                    if check_pivot_breakouts(data):
                        pivot_point_tickers.append(ticker)

            # Fetch latest data and indicators for the selected stocks
            def fetch_latest_data(tickers):
                technical_data = []
                for ticker in tickers:
                    data = yf.download(ticker, period='1y')
                    if data.empty:
                        continue
                    data['5_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=5)
                    data['15_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=15)
                    data['MACD'] = ta.trend.macd(data['Close'])
                    data['MACD_Hist'] = ta.trend.macd_diff(data['Close'])
                    data['RSI'] = ta.momentum.rsi(data['Close'])
                    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
                    data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'])
                    data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'])
                    data['20_day_vol_MA'] = data['Volume'].rolling(window=20).mean()

                    latest_data = data.iloc[-1]
                    technical_data.append({
                        'Ticker': ticker,
                        'Close': latest_data['Close'],
                        '5_day_EMA': latest_data['5_day_EMA'],
                        '15_day_EMA': latest_data['15_day_EMA'],
                        'MACD': latest_data['MACD'],
                        'MACD_Hist': latest_data['MACD_Hist'],
                        'RSI': latest_data['RSI'],
                        'ADX': latest_data['ADX'],
                        'Bollinger_High': latest_data['Bollinger_High'],
                        'Bollinger_Low': latest_data['Bollinger_Low'],
                        'Volume': latest_data['Volume'],
                        '20_day_vol_MA': latest_data['20_day_vol_MA']
                    })
                return pd.DataFrame(technical_data)

            # Generate dataframes for the selected strategies
            df_macd_signal = fetch_latest_data(macd_signal_list)
            df_negative_histogram = fetch_latest_data(negative_histogram_tickers)
            df_moving_average_signal = fetch_latest_data(moving_average_tickers)
            df_bollinger_low_cross_signal = fetch_latest_data(bollinger_low_cross_tickers)
            df_rsi_signal = fetch_latest_data(rsi_tickers)
            df_stochastic_signal = fetch_latest_data(stochastic_tickers)
            df_ichimoku_signal = fetch_latest_data(ichimoku_tickers)
            df_adx_signal = fetch_latest_data(adx_tickers)
            df_parabolic_sar_signal = fetch_latest_data(parabolic_sar_tickers)
            df_pivot_point_signal = fetch_latest_data(pivot_point_tickers)

            st.title("Stock Analysis Based on Selected Strategy")

            if submenu == "Moving Average":
                st.write("Stocks with 50-day EMA crossing above 200-day EMA in the last 5 days:")
                st.dataframe(df_moving_average_signal)

            elif submenu == "MACD":
                st.write("Stocks with Negative MACD Histogram Increasing and Price Increasing for 3 Consecutive Days:")
                st.dataframe(df_negative_histogram)

            elif submenu == "Bollinger Bands":
                st.write("Stocks with price crossing below Bollinger Low in the last 5 days:")
                st.dataframe(df_bollinger_low_cross_signal)

            elif submenu == "RSI":
                st.write("Stocks with RSI below 30:")
                st.dataframe(df_rsi_signal)

            elif submenu == "Stochastic Oscillator":
                st.write("Stocks with %K crossing above %D from below 20 in the last 5 days:")
                st.dataframe(df_stochastic_signal)

            elif submenu == "Ichimoku Cloud":
                st.write("Stocks with price above Ichimoku Cloud:")
                st.dataframe(df_ichimoku_signal)

            elif submenu == "ADX":
                st.write("Stocks with +DI crossing above -DI in the last 5 days:")
                st.dataframe(df_adx_signal)

            elif submenu == "Parabolic SAR":
                st.write("Stocks with price crossing above Parabolic SAR in the last 5 days:")
                st.dataframe(df_parabolic_sar_signal)

            elif submenu == "Pivot Points":
                st.write("Stocks with price breaking above resistance levels in the last 5 days:")
                st.dataframe(df_pivot_point_signal)

            elif submenu == "Fibonacci Retracement":
                st.subheader("Fibonacci Retracement Strategies")

                pass

            elif submenu == "Candlestick Patterns":
                st.subheader("Candlestick Patterns")

             




        


        elif choice == "Market Stats":
        # 'Market Stats' code -------------------------------------------------------------------------------------------    
            
            # List of tickers
            tickers = bse_largecap+bse_midcap+bse_smallcap

            # Function to fetch data
            def fetch_data(tickers, period='1d', interval='1m'):
                data = yf.download(tickers, period=period, interval=interval)
                return data['Close']

            # Fetch the data for daily, weekly, and monthly periods
            data_daily = fetch_data(tickers, period='1d', interval='1m')
            data_weekly = fetch_data(tickers, period='5d', interval='1d')
            data_monthly = fetch_data(tickers, period='1mo', interval='1d')

            # Drop columns with all NaN values
            data_daily.dropna(axis=1, how='all', inplace=True)
            data_weekly.dropna(axis=1, how='all', inplace=True)
            data_monthly.dropna(axis=1, how='all', inplace=True)

            # Fill missing values with forward fill
            data_daily.fillna(method='ffill', inplace=True)
            data_weekly.fillna(method='ffill', inplace=True)
            data_monthly.fillna(method='ffill', inplace=True)

            # Fill any remaining NaNs with backward fill (in case the first row is NaN)
            data_daily.fillna(method='bfill', inplace=True)
            data_weekly.fillna(method='bfill', inplace=True)
            data_monthly.fillna(method='bfill', inplace=True)

            # Calculate daily, weekly, and monthly changes
            daily_change = data_daily.iloc[-1] - data_daily.iloc[0]
            percent_change_daily = (daily_change / data_daily.iloc[0]) 

            weekly_change = data_weekly.iloc[-1] - data_weekly.iloc[0]
            percent_change_weekly = (weekly_change / data_weekly.iloc[0]) 

            monthly_change = data_monthly.iloc[-1] - data_monthly.iloc[0]
            percent_change_monthly = (monthly_change / data_monthly.iloc[0]) 

            # Create DataFrames
            df_daily = pd.DataFrame({'Ticker': data_daily.columns, 'Last Traded Price': data_daily.iloc[-1].values,
                                    '% Change': percent_change_daily.values})
            df_weekly = pd.DataFrame({'Ticker': data_weekly.columns, 'Last Traded Price': data_weekly.iloc[-1].values,
                                    '% Change': percent_change_weekly.values})
            df_monthly = pd.DataFrame({'Ticker': data_monthly.columns, 'Last Traded Price': data_monthly.iloc[-1].values,
                                    '% Change': percent_change_monthly.values})

            # Remove rows with NaN values
            df_daily.dropna(inplace=True)
            df_weekly.dropna(inplace=True)
            df_monthly.dropna(inplace=True)

            # Top 5 Gainers and Losers for daily, weekly, and monthly
            top_gainers_daily = df_daily.nlargest(5, '% Change')
            top_losers_daily = df_daily.nsmallest(5, '% Change')

            top_gainers_weekly = df_weekly.nlargest(5, '% Change')
            top_losers_weekly = df_weekly.nsmallest(5, '% Change')

            top_gainers_monthly = df_monthly.nlargest(5, '% Change')
            top_losers_monthly = df_monthly.nsmallest(5, '% Change')

            # Function to plot bar charts with gainers and losers on a single line
            def plot_bar_chart(gainers, losers, title):
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=gainers['Ticker'],
                    y=gainers['% Change'],
                    name='Gainers',
                    marker_color='lightseagreen'
                ))

                fig.add_trace(go.Bar(
                    x=losers['Ticker'],
                    y=losers['% Change'],
                    name='Losers',
                    marker_color='lightpink'
                ))

                fig.update_layout(
                    title=title,
                    xaxis_title='Ticker',
                    yaxis_title='% Change',
                    barmode='relative',
                    bargap=0.15,
                    bargroupgap=0.1,
                    yaxis=dict(tickformat='%')
                )

                st.plotly_chart(fig)

            plot_bar_chart(top_gainers_daily, top_losers_daily, 'Top 5 Daily Gainers and Losers')
            
            plot_bar_chart(top_gainers_weekly, top_losers_weekly, 'Top 5 Weekly Gainers and Losers')
            
            plot_bar_chart(top_gainers_monthly, top_losers_monthly, 'Top 5 Monthly Gainers and Losers')

        elif choice == "Stock Comparison":
            #'Stock Comparison' code---------------------------------------------------------------------------------------------------------------------------------

            # Sidebar setup
            st.sidebar.subheader('Add Tickers')

            # Sidebar for user input
            tickers = st.sidebar.multiselect("Enter Stock Symbols", options=bse_largecap+bse_midcap+bse_smallcap)

            start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2023-01-01'))
            end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

            # Load stock data
            @st.cache_data
            def load_data(ticker, start, end):
                data = yf.download(ticker, start=start, end=end)
                data.reset_index(inplace=True)
                return data

            # Load index data
            @st.cache_data
            def load_index_data(ticker, start, end):
                data = yf.download(ticker, start=start, end=end)
                data.reset_index(inplace=True)
                return data

            st.title('Stocks Comparison- Technical Analysis')

            index_ticker = "^NSEI"  # NIFTY 50 index ticker

            # Load index data
            index_data = load_index_data(index_ticker, start_date, end_date).copy()

            # Calculate technical indicators
            def calculate_technical_indicators(df, index_df):
                # Moving averages
                df['10_SMA'] = ta.trend.sma_indicator(df['Close'], window=10)
                df['20_SMA'] = ta.trend.sma_indicator(df['Close'], window=20)
                df['10_EMA'] = ta.trend.ema_indicator(df['Close'], window=10)
                df['20_EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
                df['10_WMA'] = ta.trend.wma_indicator(df['Close'], window=10)
                df['20_WMA'] = ta.trend.wma_indicator(df['Close'], window=20)

                # Volume Moving Averages
                df['5_VMA'] = df['Volume'].rolling(window=5).mean()
                df['10_VMA'] = df['Volume'].rolling(window=10).mean()
                df['20_VMA'] = df['Volume'].rolling(window=20).mean()

                # Momentum Indicators
                df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                df['%K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
                df['%D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
                df['MACD'] = ta.trend.macd(df['Close'])
                df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

                # Volume Indicators
                df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
                df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
                df['A/D Line'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])

                # Volatility Indicators
                df['BB_High'], df['BB_Middle'], df['BB_Low'] = ta.volatility.bollinger_hband(df['Close']), ta.volatility.bollinger_mavg(df['Close']), ta.volatility.bollinger_lband(df['Close'])
                df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
                df['Std Dev'] = ta.volatility.bollinger_wband(df['Close'])

                # Trend Indicators
                df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
                df['+DI'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'])
                df['-DI'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'])
                psar = pta.psar(df['High'], df['Low'], df['Close'])
                df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']
                df['Ichimoku_a'] = ta.trend.ichimoku_a(df['High'], df['Low'])
                df['Ichimoku_b'] = ta.trend.ichimoku_b(df['High'], df['Low'])
                df['Ichimoku_base'] = ta.trend.ichimoku_base_line(df['High'], df['Low'])
                df['Ichimoku_conv'] = ta.trend.ichimoku_conversion_line(df['High'], df['Low'])

                # Support and Resistance Levels
                df = find_support_resistance(df)
                df['Support_Trendline'] = calculate_trendline(df, kind='support')
                df['Resistance_Trendline'] = calculate_trendline(df, kind='resistance')
                df = pivot_points(df)

                # Price Oscillators
                df['ROC'] = ta.momentum.roc(df['Close'], window=12)
                df['DPO'] = ta.trend.dpo(df['Close'], window=20)
                df['Williams %R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)

                # Market Breadth Indicators
                df['Advances'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else 0)
                df['Declines'] = df['Close'].diff().apply(lambda x: 1 if x < 0 else 0)
                df['McClellan Oscillator'] = (df['Advances'] - df['Declines']).rolling(window=19).mean() - (df['Advances'] - df['Declines']).rolling(window=39).mean()
                df['TRIN'] = (df['Advances'] / df['Declines']) / (df['Volume'][df['Advances'] > 0].sum() / df['Volume'][df['Declines'] > 0].sum())
                df['Advance-Decline Line'] = df['Advances'].cumsum() - df['Declines'].cumsum()

                # Relative Performance Indicators
                df['Price-to-Volume Ratio'] = df['Close'] / df['Volume']
                df['Relative Strength Comparison'] = df['Close'] / index_df['Close']
                df['Performance Relative to an Index'] = df['Close'].pct_change().cumsum() - index_df['Close'].pct_change().cumsum()

                return df

            # Identify Horizontal Support and Resistance
            def find_support_resistance(df, window=20):
                df['Support'] = df['Low'].rolling(window, center=True).min()
                df['Resistance'] = df['High'].rolling(window, center=True).max()
                return df

            # Draw Trendlines
            def calculate_trendline(df, kind='support'):
                if kind == 'support':
                    prices = df['Low']
                elif kind == 'resistance':
                    prices = df['High']
                else:
                    raise ValueError("kind must be either 'support' or 'resistance'")

                indices = np.arange(len(prices))
                slope, intercept, _, _, _ = linregress(indices, prices)
                trendline = slope * indices + intercept
                return trendline

            # Calculate Pivot Points
            def pivot_points(df):
                df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
                df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
                df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
                df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
                df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
                return df

            # Function to add range buttons to the plot
            def add_range_buttons(fig):
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=14, label="14d", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True)
                    )
                )

            # Plotly visualization functions
            def plot_indicator(df, indicator, title, yaxis_title='Price', secondary_y=False):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator, yaxis="y2" if secondary_y else "y1"))
                
                if secondary_y:
                    fig.update_layout(
                        yaxis2=dict(
                            title=indicator,
                            overlaying='y',
                            side='right'
                        )
                    )
                
                fig.update_layout(title=title, xaxis_title='Date', yaxis_title=yaxis_title)
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_moving_average(df, ma_type, ticker):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='blue', opacity=0.5, yaxis='y2'))
                if ma_type == 'SMA':
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['10_SMA'], mode='lines', name='10_SMA'))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['20_SMA'], mode='lines', name='20_SMA'))
                elif ma_type == 'EMA':
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['10_EMA'], mode='lines', name='10_EMA'))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['20_EMA'], mode='lines', name='20_EMA'))
                elif ma_type == 'WMA':
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['10_WMA'], mode='lines', name='10_WMA'))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['20_WMA'], mode='lines', name='20_WMA'))
                fig.update_layout(title=f'{ticker} {ma_type} Moving Averages', xaxis_title='Date', yaxis_title='Price', yaxis2=dict(title='Volume', overlaying='y', side='right'))
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_macd(df, ticker):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], mode='lines', name='MACD Signal'))

                # Plot MACD Histogram with different colors
                macd_hist_colors = []
                for i in range(1, len(df)):
                    if df['MACD_Hist'].iloc[i] > 0:
                        color = 'green' if df['MACD_Hist'].iloc[i] > df['MACD_Hist'].iloc[i - 1] else 'lightgreen'
                    else:
                        color = 'red' if df['MACD_Hist'].iloc[i] < df['MACD_Hist'].iloc[i - 1] else 'lightcoral'
                    macd_hist_colors.append(color)

                fig.add_trace(go.Bar(x=df['Date'][1:], y=df['MACD_Hist'][1:], name='MACD Histogram', marker_color=macd_hist_colors, yaxis='y2'))

                fig.update_layout(
                    title=f'{ticker} MACD',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    yaxis2=dict(
                        title='MACD Histogram',
                        overlaying='y',
                        side='right'
                    )
                )
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_trendlines(df, ticker):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                fig.add_trace(go.Scatter(x=df['Date'], y=df['Support_Trendline'], mode='lines', name='Support Trendline', line=dict(color='green', dash='dash')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Resistance_Trendline'], mode='lines', name='Resistance Trendline', line=dict(color='red', dash='dash')))

                fig.update_layout(title=f'{ticker} Trendlines', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_fibonacci_retracement(df, ticker):
                high = df['High'].max()
                low = df['Low'].min()

                diff = high - low
                levels = [high, high - 0.236 * diff, high - 0.382 * diff, high - 0.5 * diff, high - 0.618 * diff, low]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                for level in levels:
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                            y=[level, level],
                                            mode='lines', name=f'Level {level}', line=dict(dash='dash')))

                fig.update_layout(title=f'{ticker} Fibonacci Retracement Levels', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_gann_fan_lines(df, ticker):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                # Adding Gann fan lines (simple example, for more advanced lines use a proper method)
                for i in range(1, 5):
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                            y=[df['Close'].iloc[0], df['Close'].iloc[0] + i * (df['Close'].iloc[-1] - df['Close'].iloc[0]) / 4],
                                            mode='lines', name=f'Gann Fan {i}', line=dict(dash='dash')))

                fig.update_layout(title=f'{ticker} Gann Fan Lines', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_chart_patterns(df, pattern, ticker):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                # Adding example chart patterns (simple example, for more advanced patterns use a proper method)
                pattern_data = detect_chart_patterns(df, pattern)
                if pattern_data:
                    for pattern_info in pattern_data:
                        fig.add_trace(go.Scatter(x=pattern_info['x'], y=pattern_info['y'], mode='lines+markers', name=pattern_info['name'], line=dict(color=pattern_info['color'])))

                fig.update_layout(title=f'{ticker} {pattern}', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

            def detect_chart_patterns(df, pattern):
                patterns = []
                if pattern == 'Head and Shoulders':
                    patterns = detect_head_and_shoulders(df)
                elif pattern == 'Double Tops and Bottoms':
                    patterns = detect_double_tops_and_bottoms(df)
                elif pattern == 'Flags and Pennants':
                    patterns = detect_flags_and_pennants(df)
                elif pattern == 'Triangles':
                    patterns = detect_triangles(df)
                elif pattern == 'Cup and Handle':
                    patterns = detect_cup_and_handle(df)
                return patterns

            def detect_head_and_shoulders(df):
                patterns = []
                window = 20  # Sliding window size
                for i in range(window, len(df) - window):
                    window_df = df.iloc[i - window:i + window]
                    if len(window_df) < 3:  # Ensure there are enough data points
                        continue
                    max_high = window_df['High'].max()
                    min_low = window_df['Low'].min()
                    middle_idx = window_df['High'].idxmax()
                    left_idx = window_df.iloc[:middle_idx]['High'].idxmax()
                    right_idx = window_df.iloc[middle_idx + 1:]['High'].idxmax()

                    if window_df['High'].iloc[left_idx] < max_high and window_df['High'].iloc[right_idx] < max_high and \
                            window_df['Low'].iloc[middle_idx] > min_low:
                        patterns.append({
                            "x": [window_df['Date'].iloc[left_idx], window_df['Date'].iloc[middle_idx], window_df['Date'].iloc[right_idx]],
                            "y": [window_df['High'].iloc[left_idx], window_df['High'].iloc[middle_idx], window_df['High'].iloc[right_idx]],
                            "name": "Head and Shoulders",
                            "color": "orange"
                        })
                return patterns

            def detect_double_tops_and_bottoms(df):
                patterns = []
                window = 20  # Sliding window size
                for i in range(window, len(df) - window):
                    window_df = df.iloc[i - window:i + window]
                    if len(window_df) < 3:  # Ensure there are enough data points
                        continue
                    max_high = window_df['High'].max()
                    min_low = window_df['Low'].min()
                    double_top = window_df['High'].value_counts().get(max_high, 0) > 1
                    double_bottom = window_df['Low'].value_counts().get(min_low, 0) > 1

                    if double_top:
                        patterns.append({
                            "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                            "y": [max_high, max_high],
                            "name": "Double Top",
                            "color": "red"
                        })
                    elif double_bottom:
                        patterns.append({
                            "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                            "y": [min_low, min_low],
                            "name": "Double Bottom",
                            "color": "green"
                        })
                return patterns

            def detect_flags_and_pennants(df):
                patterns = []
                window = 20  # Sliding window size
                for i in range(window, len(df) - window):
                    window_df = df.iloc[i - window:i + window]
                    if len(window_df) < 3:  # Ensure there are enough data points
                        continue
                    min_low = window_df['Low'].min()
                    max_high = window_df['High'].max()
                    flag_pattern = ((window_df['High'] - window_df['Low']) / window_df['Low']).mean() < 0.05

                    if flag_pattern:
                        patterns.append({
                            "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                            "y": [min_low, max_high],
                            "name": "Flag",
                            "color": "purple"
                        })
                return patterns

            def detect_triangles(df):
                patterns = []
                window = 20  # Sliding window size
                for i in range(window, len(df) - window):
                    window_df = df.iloc[i - window:i + window]
                    if len(window_df) < 3:  # Ensure there are enough data points
                        continue
                    max_high = window_df['High'].max()
                    min_low = window_df['Low'].min()
                    triangle_pattern = np.all(np.diff(window_df['High']) < 0) and np.all(np.diff(window_df['Low']) > 0)

                    if triangle_pattern:
                        patterns.append({
                            "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                            "y": [min_low, max_high],
                            "name": "Triangle",
                            "color": "blue"
                        })
                return patterns

            def detect_cup_and_handle(df):
                patterns = []
                window = 50  # Sliding window size
                for i in range(window, len(df) - window):
                    window_df = df.iloc[i - window:i + window]
                    if len(window_df) < 3:  # Ensure there are enough data points
                        continue
                    max_high = window_df['High'].max()
                    min_low = window_df['Low'].min()
                    cup_shape = ((window_df['High'] - window_df['Low']) / window_df['Low']).mean() < 0.1

                    if cup_shape:
                        patterns.append({
                            "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[len(window_df) // 2], window_df['Date'].iloc[-1]],
                            "y": [max_high, min_low, max_high],
                            "name": "Cup and Handle",
                            "color": "brown"
                        })
                return patterns

            def plot_mcclellan_oscillator(df, ticker):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['McClellan Oscillator'], mode='lines', name='McClellan Oscillator'))
                fig.update_layout(title=f'{ticker} McClellan Oscillator', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def plot_trin(df, ticker):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['TRIN'], mode='lines', name='TRIN'))
                fig.update_layout(title=f'{ticker} Arms Index (TRIN)', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            def get_signals(df):
                signals = []

                # Example logic for signals (these can be customized)
                if df['Close'].iloc[-1] > df['20_SMA'].iloc[-1]:
                    signals.append(("Simple Moving Average (20_SMA)", "Hold", "Price is above the SMA."))
                else:
                    signals.append(("Simple Moving Average (20_SMA)", "Sell", "Price crossed below the SMA."))

                if df['Close'].iloc[-1] > df['20_EMA'].iloc[-1]:
                    signals.append(("Exponential Moving Average (20_EMA)", "Hold", "Price is above the EMA."))
                else:
                    signals.append(("Exponential Moving Average (20_EMA)", "Sell", "Price crossed below the EMA."))

                if df['Close'].iloc[-1] > df['20_WMA'].iloc[-1]:
                    signals.append(("Weighted Moving Average (20_WMA)", "Hold", "Price is above the WMA."))
                else:
                    signals.append(("Weighted Moving Average (20_WMA)", "Sell", "Price crossed below the WMA."))

                if df['RSI'].iloc[-1] < 30:
                    signals.append(("Relative Strength Index (RSI)", "Buy", "RSI crosses below 30 (oversold)."))
                elif df['RSI'].iloc[-1] > 70:
                    signals.append(("Relative Strength Index (RSI)", "Sell", "RSI crosses above 70 (overbought)."))
                else:
                    signals.append(("Relative Strength Index (RSI)", "Hold", "RSI is between 30 and 70."))

                if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                    signals.append(("Moving Average Convergence Divergence (MACD)", "Buy", "MACD line crosses above the signal line."))
                else:
                    signals.append(("Moving Average Convergence Divergence (MACD)", "Sell", "MACD line crosses below the signal line."))

                if df['%K'].iloc[-1] < 20 and df['%D'].iloc[-1] < 20 and df['%K'].iloc[-1] > df['%D'].iloc[-1]:
                    signals.append(("Stochastic Oscillator", "Buy", "%K line crosses above %D line and both are below 20."))
                elif df['%K'].iloc[-1] > 80 and df['%D'].iloc[-1] > 80 and df['%K'].iloc[-1] < df['%D'].iloc[-1]:
                    signals.append(("Stochastic Oscillator", "Sell", "%K line crosses below %D line and both are above 80."))
                else:
                    signals.append(("Stochastic Oscillator", "Hold", "No clear buy or sell signal."))

                if df['OBV'].diff().iloc[-1] > 0:
                    signals.append(("On-Balance Volume (OBV)", "Buy", "OBV is increasing."))
                else:
                    signals.append(("On-Balance Volume (OBV)", "Sell", "OBV is decreasing."))

                if df['Close'].iloc[-1] > df['VWAP'].iloc[-1]:
                    signals.append(("Volume Weighted Average Price (VWAP)", "Buy", "Price crosses above the VWAP."))
                else:
                    signals.append(("Volume Weighted Average Price (VWAP)", "Sell", "Price crosses below the VWAP."))

                if df['A/D Line'].diff().iloc[-1] > 0:
                    signals.append(("Accumulation/Distribution Line (A/D Line)", "Buy", "A/D Line is increasing."))
                else:
                    signals.append(("Accumulation/Distribution Line (A/D Line)", "Sell", "A/D Line is decreasing."))

                if df['Close'].iloc[-1] < df['BB_Low'].iloc[-1]:
                    signals.append(("Bollinger Bands", "Buy", "Price crosses below the lower band."))
                elif df['Close'].iloc[-1] > df['BB_High'].iloc[-1]:
                    signals.append(("Bollinger Bands", "Sell", "Price crosses above the upper band."))
                else:
                    signals.append(("Bollinger Bands", "Hold", "Price is within Bollinger Bands."))

                if df['ATR'].iloc[-1] > df['ATR'].rolling(window=14).mean().iloc[-1]:
                    signals.append(("Average True Range (ATR)", "Buy", "ATR is increasing, indicating higher volatility."))
                else:
                    signals.append(("Average True Range (ATR)", "Sell", "ATR is decreasing, indicating lower volatility."))

                if df['Std Dev'].iloc[-1] > df['Close'].rolling(window=20).std().iloc[-1]:
                    signals.append(("Standard Deviation", "Buy", "Price is below the mean minus 2 standard deviations."))
                else:
                    signals.append(("Standard Deviation", "Sell", "Price is above the mean plus 2 standard deviations."))

                if df['Parabolic_SAR'].iloc[-1] < df['Close'].iloc[-1]:
                    signals.append(("Parabolic SAR (Stop and Reverse)", "Buy", "Price crosses above the SAR."))
                else:
                    signals.append(("Parabolic SAR (Stop and Reverse)", "Sell", "Price crosses below the SAR."))

                if df['ROC'].iloc[-1] > 0:
                    signals.append(("Price Rate of Change (ROC)", "Buy", "ROC crosses above zero."))
                else:
                    signals.append(("Price Rate of Change (ROC)", "Sell", "ROC crosses below zero."))

                if df['DPO'].iloc[-1] > 0:
                    signals.append(("Detrended Price Oscillator (DPO)", "Buy", "DPO crosses above zero."))
                else:
                    signals.append(("Detrended Price Oscillator (DPO)", "Sell", "DPO crosses below zero."))

                if df['Williams %R'].iloc[-1] < -80:
                    signals.append(("Williams %R", "Buy", "Williams %R crosses above -80 (indicating oversold)."))
                elif df['Williams %R'].iloc[-1] > -20:
                    signals.append(("Williams %R", "Sell", "Williams %R crosses below -20 (indicating overbought)."))
                else:
                    signals.append(("Williams %R", "Hold", "Williams %R is between -80 and -20."))

                if df['Close'].iloc[-1] > df['Pivot'].iloc[-1]:
                    signals.append(("Pivot Points", "Buy", "Price crosses above the pivot point."))
                else:
                    signals.append(("Pivot Points", "Sell", "Price crosses below the pivot point."))

                high = df['High'].max()
                low = df['Low'].min()
                diff = high - low
                fib_levels = [high, high - 0.236 * diff, high - 0.382 * diff, high - 0.5 * diff, high - 0.618 * diff, low]
                for level in fib_levels:
                    if df['Close'].iloc[-1] > level:
                        signals.append(("Fibonacci Retracement Levels", "Buy", "Price crosses above a Fibonacci retracement level."))
                        break
                    elif df['Close'].iloc[-1] < level:
                        signals.append(("Fibonacci Retracement Levels", "Sell", "Price crosses below a Fibonacci retracement level."))
                        break

                gann_fan_line = [df['Close'].iloc[0] + i * (df['Close'].iloc[-1] - df['Close'].iloc[0]) / 4 for i in range(1, 5)]
                for line in gann_fan_line:
                    if df['Close'].iloc[-1] > line:
                        signals.append(("Gann Fan Lines", "Buy", "Price crosses above a Gann fan line."))
                        break
                    elif df['Close'].iloc[-1] < line:
                        signals.append(("Gann Fan Lines", "Sell", "Price crosses below a Gann fan line."))
                        break

                if df['McClellan Oscillator'].iloc[-1] > 0:
                    signals.append(("McClellan Oscillator", "Buy", "Oscillator crosses above zero."))
                else:
                    signals.append(("McClellan Oscillator", "Sell", "Oscillator crosses below zero."))

                if df['TRIN'].iloc[-1] < 1:
                    signals.append(("Arms Index (TRIN)", "Buy", "TRIN below 1.0 (more advancing volume)."))
                else:
                    signals.append(("Arms Index (TRIN)", "Sell", "TRIN above 1.0 (more declining volume)."))

                # Chart Patterns
                patterns = detect_chart_patterns(df, 'Summary')
                signals.extend(patterns)

                # Additional Indicators
                if df['Ichimoku_a'].iloc[-1] > df['Ichimoku_b'].iloc[-1]:
                    signals.append(("Ichimoku Cloud", "Buy", "Ichimoku conversion line above baseline."))
                else:
                    signals.append(("Ichimoku Cloud", "Sell", "Ichimoku conversion line below baseline."))

                if df['Relative Strength Comparison'].iloc[-1] > 1:
                    signals.append(("Relative Strength Comparison", "Buy", "Stock outperforms index."))
                else:
                    signals.append(("Relative Strength Comparison", "Sell", "Stock underperforms index."))

                if df['Performance Relative to an Index'].iloc[-1] > 0:
                    signals.append(("Performance Relative to an Index", "Buy", "Stock outperforms index over time."))
                else:
                    signals.append(("Performance Relative to an Index", "Sell", "Stock underperforms index over time."))

                if df['Advance-Decline Line'].diff().iloc[-1] > 0:
                    signals.append(("Advance-Decline Line", "Buy", "Advances exceed declines."))
                else:
                    signals.append(("Advance-Decline Line", "Sell", "Declines exceed advances."))

                if df['Price-to-Volume Ratio'].iloc[-1] > df['Price-to-Volume Ratio'].rolling(window=14).mean().iloc[-1]:
                    signals.append(("Price-to-Volume Ratio", "Buy", "Price-to-Volume ratio increasing."))
                else:
                    signals.append(("Price-to-Volume Ratio", "Sell", "Price-to-Volume ratio decreasing."))

                return signals

            # Sidebar for technical indicators
            st.sidebar.header('Technical Indicators')
            indicator_category = st.sidebar.radio('Select Indicator Category', [
                'Moving Averages', 'Momentum Indicators', 'Volume Indicators', 'Volatility Indicators', 'Trend Indicators',
                'Support and Resistance Levels', 'Price Oscillators', 'Market Breadth Indicators', 'Chart Patterns', 'Relative Performance Indicators', 'Summary'
            ])

            # Display technical indicators
            st.subheader('Technical Indicators')
            for ticker in tickers:
                data = load_data(ticker, start_date, end_date).copy()
                data = calculate_technical_indicators(data, index_data)

                if indicator_category != 'Summary':
                    if indicator_category == 'Moving Averages':
                        indicators = st.selectbox("Select Moving Average", ['SMA', 'EMA', 'WMA'], key=f'{ticker}_ma')
                        plot_moving_average(data, indicators, ticker)
                    elif indicator_category == 'Momentum Indicators':
                        indicators = st.selectbox("Select Momentum Indicator", ['RSI', 'Stochastic Oscillator', 'MACD'], key=f'{ticker}_momentum')
                        if indicators == 'RSI':
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=[70] * len(data), mode='lines', name='RSI 70', line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=data['Date'], y=[30] * len(data), mode='lines', name='RSI 30', line=dict(color='green', dash='dash')))
                            fig.update_layout(title=f'{ticker} Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='Value')
                            add_range_buttons(fig)
                            st.plotly_chart(fig)
                        elif indicators == 'Stochastic Oscillator':
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['%K'], mode='lines', name='%K'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['%D'], mode='lines', name='%D'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=[80] * len(data), mode='lines', name='%K 80', line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=data['Date'], y=[20] * len(data), mode='lines', name='%K 20', line=dict(color='green', dash='dash')))
                            fig.update_layout(title=f'{ticker} Stochastic Oscillator', xaxis_title='Date', yaxis_title='Value')
                            add_range_buttons(fig)
                            st.plotly_chart(fig)
                        elif indicators == 'MACD':
                            plot_macd(data, ticker)
                    elif indicator_category == 'Volume Indicators':
                        indicators = st.selectbox("Select Volume Indicator", ['OBV', 'VWAP', 'A/D Line', 'Volume Moving Averages'], key=f'{ticker}_volume')
                        if indicators == 'OBV':
                            plot_indicator(data, 'OBV', f'{ticker} On-Balance Volume (OBV)')
                        elif indicators == 'VWAP':
                            plot_indicator(data, 'VWAP', f'{ticker} Volume Weighted Average Price (VWAP)')
                        elif indicators == 'A/D Line':
                            plot_indicator(data, 'A/D Line', f'{ticker} Accumulation/Distribution Line')
                        elif indicators == 'Volume Moving Averages':
                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume', marker_color='blue', opacity=0.5))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['5_VMA'], mode='lines', name='5_VMA'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['10_VMA'], mode='lines', name='10_VMA'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['20_VMA'], mode='lines', name='20_VMA'))
                            fig.update_layout(title=f'{ticker} Volume Moving Averages', xaxis_title='Date', yaxis_title='Volume')
                            add_range_buttons(fig)
                            st.plotly_chart(fig)
                    elif indicator_category == 'Volatility Indicators':
                        indicators = st.selectbox("Select Volatility Indicator", ['Bollinger Bands', 'ATR', 'Standard Deviation'], key=f'{ticker}_volatility')
                        if indicators == 'Bollinger Bands':
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_High'], mode='lines', name='BB High'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Middle'], mode='lines', name='BB Middle'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Low'], mode='lines', name='BB Low'))
                            fig.update_layout(title=f'{ticker} Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
                            add_range_buttons(fig)
                            st.plotly_chart(fig)
                        elif indicators == 'ATR':
                            plot_indicator(data, 'ATR', f'{ticker} Average True Range (ATR)')
                        elif indicators == 'Standard Deviation':
                            plot_indicator(data, 'Std Dev', f'{ticker} Standard Deviation')
                    elif indicator_category == 'Trend Indicators':
                        indicators = st.selectbox("Select Trend Indicator", ['Trendlines', 'Parabolic SAR', 'Ichimoku Cloud', 'ADX'], key=f'{ticker}_trend')
                        if indicators == 'Trendlines':
                            plot_trendlines(data, ticker)
                        elif indicators == 'Parabolic SAR':
                            plot_indicator(data, 'Parabolic_SAR', f'{ticker} Parabolic SAR')
                        elif indicators == 'Ichimoku Cloud':
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_a'], mode='lines', name='Ichimoku A'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_b'], mode='lines', name='Ichimoku B'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_base'], mode='lines', name='Ichimoku Base Line'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_conv'], mode='lines', name='Ichimoku Conversion Line'))
                            fig.update_layout(title=f'{ticker} Ichimoku Cloud', xaxis_title='Date', yaxis_title='Value')
                            add_range_buttons(fig)
                            st.plotly_chart(fig)
                        elif indicators == 'ADX':
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['ADX'], mode='lines', name='ADX'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['+DI'], mode='lines', name='+DI'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['-DI'], mode='lines', name='-DI'))
                            fig.update_layout(title=f'{ticker} Average Directional Index (ADX)', xaxis_title='Date', yaxis_title='Value')
                            add_range_buttons(fig)
                            st.plotly_chart(fig)
                    elif indicator_category == 'Support and Resistance Levels':
                        indicators = st.selectbox("Select Support and Resistance Level", ['Pivot Points', 'Fibonacci Retracement Levels', 'Gann Fan Lines'], key=f'{ticker}_support_resistance')
                        if indicators == 'Pivot Points':
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['Pivot'], mode='lines', name='Pivot'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['R1'], mode='lines', name='R1'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['S1'], mode='lines', name='S1'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['R2'], mode='lines', name='R2'))
                            fig.add_trace(go.Scatter(x=data['Date'], y=data['S2'], mode='lines', name='S2'))
                            fig.update_layout(title=f'{ticker} Pivot Points', xaxis_title='Date', yaxis_title='Price')
                            add_range_buttons(fig)
                            st.plotly_chart(fig)
                        elif indicators == 'Fibonacci Retracement Levels':
                            plot_fibonacci_retracement(data, ticker)
                        elif indicators == 'Gann Fan Lines':
                            plot_gann_fan_lines(data, ticker)
                    elif indicator_category == 'Price Oscillators':
                        indicators = st.selectbox("Select Price Oscillator", ['ROC', 'DPO', 'Williams %R'], key=f'{ticker}_oscillators')
                        if indicators == 'ROC':
                            plot_indicator(data, 'ROC', f'{ticker} Rate of Change (ROC)')
                        elif indicators == 'DPO':
                            plot_indicator(data, 'DPO', f'{ticker} Detrended Price Oscillator (DPO)')
                        elif indicators == 'Williams %R':
                            plot_indicator(data, 'Williams %R', f'{ticker} Williams %R')
                    elif indicator_category == 'Market Breadth Indicators':
                        indicators = st.selectbox("Select Market Breadth Indicator", ['Advance-Decline Line', 'McClellan Oscillator', 'TRIN'], key=f'{ticker}_breadth')
                        if indicators == 'Advance-Decline Line':
                            plot_indicator(data, 'Advance-Decline Line', f'{ticker} Advance-Decline Line')
                        elif indicators == 'McClellan Oscillator':
                            plot_mcclellan_oscillator(data, ticker)
                        elif indicators == 'TRIN':
                            plot_trin(data, ticker)
                    elif indicator_category == 'Chart Patterns':
                        indicators = st.selectbox("Select Chart Pattern", ['Head and Shoulders', 'Double Tops and Bottoms', 'Flags and Pennants', 'Triangles', 'Cup and Handle'], key=f'{ticker}_patterns')
                        if indicators == 'Head and Shoulders':
                            plot_chart_patterns(data, 'Head and Shoulders', ticker)
                        elif indicators == 'Double Tops and Bottoms':
                            plot_chart_patterns(data, 'Double Tops and Bottoms', ticker)
                        elif indicators == 'Flags and Pennants':
                            plot_chart_patterns(data, 'Flags and Pennants', ticker)
                        elif indicators == 'Triangles':
                            plot_chart_patterns(data, 'Triangles', ticker)
                        elif indicators == 'Cup and Handle':
                            plot_chart_patterns(data, 'Cup and Handle', ticker)
                    elif indicator_category == 'Relative Performance Indicators':
                        indicators = st.selectbox("Select Relative Performance Indicator", ['Price-to-Volume Ratio', 'Relative Strength Comparison', 'Performance Relative to an Index'], key=f'{ticker}_relative_performance')
                        if indicators == 'Price-to-Volume Ratio':
                            plot_indicator(data, 'Price-to-Volume Ratio', f'{ticker} Price-to-Volume Ratio', secondary_y=True)
                        elif indicators == 'Relative Strength Comparison':
                            plot_indicator(data, 'Relative Strength Comparison', f'{ticker} Relative Strength Comparison')
                        elif indicators == 'Performance Relative to an Index':
                            plot_indicator(data, 'Performance Relative to an Index', f'{ticker} Performance Relative to an Index')
                else:
                    # Display signals in a dataframe with improved visualization
                    st.subheader(f'{ticker} Technical Indicator Signals')
                    signals = get_signals(data)
                    signals_df = pd.DataFrame(signals, columns=['Technical Indicator', 'Signal', 'Reason'])
                    st.dataframe(signals_df.style.applymap(lambda x: 'background-color: lightgreen' if 'Buy' in x else 'background-color: lightcoral' if 'Sell' in x else '', subset=['Signal']))
