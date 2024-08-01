import os
import streamlit as st
import random
import string
from datetime import datetime, timedelta
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
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import pandas_ta as pta
import numpy as np
import plotly.graph_objs as go
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots




# technical analysis
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta



# news
import streamlit as st
from newsapi.newsapi_client import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# time series 
import yfinance as yf
import pandas as pd
import numpy as np
import itertools
from ta import add_all_ta_features
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.graph_objects as go
from pmdarima import auto_arima
from datetime import timedelta, datetime
from scipy.signal import cwt, ricker, hilbert


#Database
# from get_tables import display_tables
from mongodb import users_collection, watchlists_collection, portfolios_collection, init_db
from bson.objectid import ObjectId

# Initialize MongoDB collections
init_db()



# List of stock tickers
bse_largecap = ["ABB.BO","ADANIENSOL.BO","ADANIENT.BO","ADANIGREEN.BO","ADANIPORTS.BO","ADANIPOWER.BO","ATGL.NS","AWL.NS","AMBUJACEM.BO","APOLLOHOSP.BO","ASIANPAINT.BO","DMART.BO","AXISBANK.BO","BAJAJ-AUTO.BO","BAJFINANCE.BO","BAJAJFINSV.BO","BAJAJHLDNG.BO","BANDHANBNK.BO","BANKBARODA.BO","BERGEPAINT.BO","BEL.BO","BPCL.BO","BHARTIARTL.BO","BOSCHLTD.BO","BRITANNIA.BO","CHOLAFIN.BO","CIPLA.BO","COALINDIA.BO","DABUR.BO","DIVISLAB.BO","DLF.BO","DRREDDY.BO","EICHERMOT.BO","NYKAA.NS","GAIL.BO","GODREJCP.BO","GRASIM.BO","HAVELLS.BO","HCLTECH.BO","HDFCAMC.BO","HDFCBANK.BO","HDFCLIFE.BO","HEROMOTOCO.BO","HINDALCO.BO","HAL.BO","HINDUNILVR.BO","HINDZINC.BO","ICICIBANK.BO","ICICIGI.BO","ICICIPRULI.BO","IOC.BO","INDUSTOWER.BO","INDUSINDBK.BO","NAUKRI.BO","INFY.BO","INDIGO.BO","ITC.BO","JIOFIN.NS","JSWSTEEL.BO","KOTAKBANK.BO","LT.BO","LICI.NS","LTIM.BO","M&M.BO","MANKIND.NS","MARICO.BO","MARUTI.BO","NESTLEIND.BO","NTPC.BO","ONGC.BO","PAYTM.NS","PIDILITIND.BO","POWERGRID.BO","PNB.BO","RELIANCE.BO","SBICARD.BO","SBILIFE.BO","SHREECEM.BO","SIEMENS.BO","SRF.BO","SBIN.BO","SUNPHARMA.BO","TCS.BO","TATACONSUM.BO","TATAMOTORS.BO","TATAMTRDVR.BO","TATAPOWER.BO","TATASTEEL.BO","TECHM.BO","TITAN.BO","ULTRACEMCO.BO","UNITDSPR.BO","UPL.BO","VBL.BO","VEDL.BO","WIPRO.BO","ZOMATO.BO","ZYDUSLIFE.NS"]
bse_midcap = ["3MINDIA.BO","AARTIIND.BO","ABBOTINDIA.BO","ACC.BO","ABCAPITAL.BO","ABFRL.BO","AJANTPHARM.BO","ALKEM.BO","APLAPOLLO.BO","ASHOKLEY.BO","ASTRAL.BO","AUBANK.BO","AUROPHARMA.BO","BALKRISIND.BO","BANKINDIA.BO","BAYERCROP.BO","BHARATFORG.BO","BHEL.BO","BIOCON.BO","CANBK.BO","CASTROLIND.BO","CGPOWER.BO","CLEAN.BO","COLPAL.BO","CONCOR.BO","COROMANDEL.BO","CRISIL.BO","CROMPTON.BO","CUMMINSIND.BO","DALBHARAT.BO","DEEPAKNTR.BO","DELHIVERY.NS","EMAMILTD.BO","ENDURANCE.BO","EXIDEIND.BO","FEDERALBNK.BO","GICRE.BO","GILLETTE.BO","GLAND.BO","GLAXO.BO","GLENMARK.BO","GMRINFRA.BO","GODREJIND.BO","GODREJPROP.BO","FLUOROCHEM.BO","GUJGASLTD.BO","HINDPETRO.BO","HONAUT.BO","ISEC.BO","IDBI.BO","IDFCFIRSTB.BO","INDIANB.BO","INDHOTEL.BO","IOB.BO","IRCTC.BO","IRFC.BO","IREDA.BO","IGL.BO","IPCALAB.BO","JINDALSTEL.BO","JSWENERGY.BO","JSWINFRA.NS","JUBLFOOD.BO","KANSAINER.BO","LTF.BO","LTTS.BO","LAURUSLABS.BO","LICHSGFIN.BO","LINDEINDIA.BO","LUPIN.BO","LODHA.BO","M&MFIN.BO","MFSL.BO","MAXHEALTH.BO","MPHASIS.BO","MRF.BO","MUTHOOTFIN.BO","NHPC.BO","NAM-INDIA.BO","NMDC.BO","NUVOCO.BO","OBEROIRLTY.BO","OIL.BO","OFSS.BO","PAGEIND.BO","PATANJALI.NS","PAYTM.BO","PERSISTENT.BO","PETRONET.BO","PIIND.BO","PEL.BO","POLYCAB.BO","PFC.BO","PGHH.BO","RAJESHEXPO.BO","RAMCOCEM.BO","RECLTD.BO","RELAXO.BO","MOTHERSON.BO","SCHAEFFLER.BO","SHRIRAMFIN.BO","SJVN.BO","SOLARINDS.BO","SONACOMS.BO","STARHEALTH.NS","SAIL.BO","SUNTV.BO","SUPREMEIND.BO","TATACOMM.BO","TATAELXSI.BO","TATATECH.BO","NIACL.BO","TORNTPHARM.BO","TORNTPOWER.BO","TRENT.BO","TIINDIA.BO","TVSMOTOR.BO","UCOBANK.BO","UNIONBANK.BO","UBL.BO","UNOMINDA.BO","VEDL.BO","IDEA.BO","VOLTAS.BO","WHIRLPOOL.BO","YESBANK.BO","ZEEL.BO"]
bse_smallcap = [
    '360ONE.NS', '3IINFOLTD.NS', '5PAISA.NS', '63MOONS.NS', 'AARTIDRUGS.NS', 'AARTIPHARM.NS', 'AAVAS.NS', 'AHL.NS',
    'ACCELYA.NS', 'ACE.NS', 'ADFFOODS.NS', 'ABSLAMC.NS', 'AVL.BO', 'ADORWELD.NS', 'ADVENZYMES.NS', 'AEGISCHEM.NS',
    'AEROFLEX.NS', 'AETHER.NS', 'AFFLE.NS', 'AGARIND.NS', 'AGI.NS', 'ATFL.NS', 'AGSTRA.NS', 'AHLUCONT.NS', 'AIAENG.NS',
    'AJMERA.NS', 'AKZOINDIA.NS', 'ALEMBICLTD.NS', 'APLLTD.NS', 'ALICON.NS', 'ALKYLAMINE.NS', 'ACLGATI.NS', 'ALLCARGO.NS',
    'ATL.NS', 'ALLSEC.NS', 'ALOKINDS.NS', 'ARE&M.NS', 'AMBER.NS', 'AMBIKCO.NS', 'AMIORG.NS', 'ANANDRATHI.NS', 'ANANTRAJ.NS',
    'ANDHRAPAP.NS', 'ANDHRAPET.BO', 'ANDREWYU.NS', 'ANGELONE.NS', 'AWHCL.NS', 'ANURAS.NS', 'APARINDS.NS', 'APCOTEXIND.NS',
    'APOLLO.NS', 'APOLLOPIPE.NS', 'APOLLOTYRE.NS', 'APTECHT.NS', 'APTUS.NS', 'ACI.NS', 'ARIHANTCAP.NS', 'ARIHANTSUP.NS',
    'ARMANFIN.NS', 'ARTEMISMED.NS', 'ARVINDFASN.NS', 'ARVIND.NS', 'ARVSMART.NS', 'ASAHIINDIA.NS', 'ASHAPURMIN.NS',
    'ASHIANA.NS', 'ASHOKA.NS', 'ASIANTILES.NS', 'ASKAUTOLTD.NS', 'ASALCBR.NS', 'ASTEC.NS', 'ASTERDM.NS', 'ASTRAMICRO.NS',
    'ASTRAZEN.NS', 'ATULAUTO.NS', 'ATUL.NS', 'AURIONPRO.NS', 'AIIL.BO', 'AUTOAXLES.NS', 'ASAL.NS', 'AVADHSUGAR.NS', 'AVALON.NS',
    'AVANTEL.NS', 'AVANTIFEED.NS', 'AVTNPL.NS', 'AXISCADES.NS', 'AZAD.NS', 'BLKASHYAP.NS', 'BAJAJCON.NS', 'BAJAJELEC.NS',
    'BAJAJHCARE.NS', 'BAJAJHIND.NS', 'BALAMINES.NS', 'BALMLAWRIE.NS', 'BLIL.BO', 'BALRAMCHIN.NS', 'BALUFORGE.BO',
    'BANCOINDIA.NS', 'MAHABANK.NS', 'BANARISUG.NS', 'BARBEQUE.NS', 'BASF.NS', 'BATAINDIA.NS', 'BCLIND.NS', 'BLAL.NS', 'BEML.NS',
    'BESTAGRO.NS', 'BFINVEST.NS', 'BFUTILITIE.NS', 'BHAGERIA.NS', 'BHAGCHEM.NS', 'BEPL.NS', 'BBL.NS', 'BDL.NS', 'BHARATWIRE.NS',
    'BIGBLOC.NS', 'BIKAJI.NS', 'BIRLACORPN.NS', 'BSOFT.NS', 'BBOX.NS', 'BLACKROSE.BO', 'BLISSGVS.NS', 'BLS.NS', 'BLUEDART.NS',
    'BLUEJET.NS', 'BLUESTARCO.NS', 'BODALCHEM.NS', 'BOMDYEING.NS', 'BOROLTD.NS', 'BORORENEW.NS', 'BRIGADE.NS', 'BUTTERFLY.NS',
    'MAPMYINDIA.NS', 'CAMLINFINE.NS', 'CAMPUS.NS', 'CANFINHOME.NS', 'CANTABIL.NS', 'CAPACITE.NS', 'CAPLIPOINT.NS', 'CGCL.NS',
    'CARBORUNIV.NS', 'CARERATING.NS', 'CARTRADE.NS', 'CARYSIL.NS', 'CCL.NS', 'CEATLTD.NS', 'CELLO.NS', 'CENTRALBK.NS',
    'CENTRUM.NS', 'CENTUM.NS', 'CENTENKA.NS', 'CENTURYPLY.NS', 'CENTURYTEX.NS', 'CERA.NS', 'CESC.NS', 'CHALET.NS', 'CLSEL.NS',
    'CHAMBLFERT.NS', 'CHEMCON.NS', 'CHEMPLASTS.NS', 'CHENNPETRO.NS', 'CHEVIOT.NS', 'CHOICEIN.NS', 'CHOLAHLDNG.NS', 'CIEINDIA.NS',
    'CIGNITITEC.NS', 'CUB.NS', 'CMSINFO.NS', 'COCHINSHIP.NS', 'COFFEEDAY.NS', 'COFORGE.NS', 'CAMS.NS', 'CONCORDBIO.NS',
    'CONFIPET.NS', 'CONTROLPR.NS', 'COSMOFIRST.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'MUFTI.BO', 'CRESSAN.BO', 'CSBBANK.NS',
    'CYIENTDLM.NS', 'CYIENT.NS', 'DLINKINDIA.NS', 'DALMIASUG.NS', 'DATAPATTNS.NS', 'DATAMATICS.NS', 'DBCORP.NS', 'DCBBANK.NS',
    'DCMSHRIRAM.NS', 'DCW.NS', 'DCXINDIA.NS', 'DDEVPLASTIK.BO', 'DECCANCE.NS', 'DEEPINDS.NS', 'DEEPAKFERT.NS', 'DELTACORP.NS',
    'DEN.NS', 'DEVYANI.NS', 'DBOL.NS', 'DHAMPURSUG.NS', 'DHANI.NS', 'DHANUKA.NS', 'DMCC.NS', 'DHARMAJ.NS', 'DHUNINV.NS', 'DBL.NS',
    'DISHTV.NS', 'DCAL.NS', 'DIVGIITTS.NS', 'DIXON.NS', 'DODLA.NS', 'DOLATALGO.NS', 'DOLLAR.NS', 'DOMS.NS', 'LALPATHLAB.NS',
    'DREAMFOLKS.NS', 'DREDGECORP.NS', 'DWARKESH.NS', 'DYNAMATECH.NS', 'EASEMYTRIP.NS', 'ECLERX.NS', 'EDELWEISS.NS', 'EIDPARRY.NS',
    'EIHAHOTELS.NS', 'EIHOTEL.NS', 'EKIEGS.NS', 'ELECON.NS', 'EMIL.NS', 'ELECTCAST.NS', 'ELGIEQUIP.NS', 'ELIN.NS',
    'ELPROINTL.BO', 'EMAMIPAP.NS', 'EMSLIMITED.NS', 'EMUDHRA.NS', 'ENGINERSIN.NS', 'ENIL.NS', 'EPIGRAL.NS', 'EPL.NS',
    'EQUITASBNK.NS', 'ERIS.NS', 'ESABINDIA.NS', 'ESAFSFB.NS', 'ESCORTS.NS', 'ESTER.NS', 'ETHOSLTD.NS', 'EUREKAFORBE.BO',
    'EVEREADY.NS', 'EVERESTIND.NS', 'EKC.NS', 'EXCELINDUS.NS', 'EXPLEOSOL.NS', 'FAIRCHEMOR.NS', 'FAZE3Q.BO', 'FDC.NS',
    'FEDFINA.NS', 'FMGOETZE.NS', 'FIEMIND.NS', 'FILATEX.NS', 'FINEORG.NS', 'FCL.NS', 'FINOPB.NS', 'FINCABLES.NS', 'FINPIPE.NS',
    'FSL.NS', 'FIVESTAR.NS', 'FLAIR.NS', 'FOODSIN.NS', 'FORCEMOT.NS', 'FORTIS.NS', 'FOSECOIND.NS', 'FUSION.NS', 'GMBREW.NS',
    'GNA.NS', 'GRINFRA.NS', 'GABRIEL.NS', 'GALAXYSURF.NS', 'GALLANTT.NS', 'GANDHAR.NS', 'GANDHITUBE.NS', 'GANESHBE.NS',
    'GANESHHOUC.NS', 'GANECOS.NS', 'GRSE.NS', 'GRWRHITECH.NS', 'GARFIBRES.NS', 'GATEWAY.NS', 'GEPIL.NS', 'GET&D.NS',
    'GENESYS.NS', 'GENSOL.NS', 'GENUSPOWER.NS', 'GEOJITFSL.NS', 'GFLLIMITED.NS', 'GHCL.NS', 'GHCLTEXTIL.NS', 'GICHSGFIN.NS',
    'GLENMARK.NS', 'MEDANTA.NS', 'GSLSU.NS', 'GLOBUSSPR.NS', 'GMMPFAUDLR.NS', 'GMRINFRA.NS', 'GOCOLORS.NS', 'GOCLCORP.NS',
    'GPIL.NS', 'GODFRYPHLP.NS', 'GODREJAGRO.NS', 'GOKEX.NS', 'GOKUL.NS', 'GOLDIAM.NS', 'GOODLUCK.NS', 'GOODYEAR.NS',
    'GRANULES.NS', 'GRAPHITE.NS', 'GRAUWEIL.NS', 'GRAVITA.NS', 'GESHIP.NS', 'GREAVESCOT.NS', 'GREENLAM.NS', 'GREENPANEL.NS',
    'GREENPLY.NS', 'GRINDWELL.NS', 'GRMOVER.NS', 'GTLINFRA.NS', 'GTPL.NS', 'GUFICBIO.NS', 'GUJALKALI.NS', 'GAEL.NS', 'GIPCL.NS',
    'GMDCLTD.NS', 'GNFC.NS', 'GPPL.NS', 'GSFC.NS', 'GSPL.NS', 'GUJTHEM.BO', 'GULFOILLUB.NS', 'GULPOLY.NS', 'HGINFRA.NS',
    'HAPPSTMNDS.NS', 'HAPPYFORGE.NS', 'HARDWYN.NS', 'HARIOMPIPE.NS', 'HARSHA.NS', 'HATHWAY.NS', 'HATSUN.NS', 'HAWKINCOOK.BO',
    'HBLPOWER.NS', 'HCG.NS', 'HEG.NS', 'HEIDELBERG.NS', 'HEMIPROP.NS', 'HERANBA.NS', 'HERCULES.NS', 'HERITGFOOD.NS',
    'HESTERBIO.NS', 'HEUBACHIND.NS', 'HFCL.NS', 'HITECH.NS', 'HIKAL.NS', 'HIL.NS', 'HSCL.NS', 'HIMATSEIDE.NS', 'HGS.NS',
    'HCC.NS', 'HINDCOPPER.NS', 'HNDFDS.NS', 'HINDOILEXP.NS', 'HINDWAREAP.NS', 'POWERINDIA.NS', 'HLEGLAS.NS', 'HLVLTD.NS',
    'HMAAGRO.NS', 'HOMEFIRST.NS', 'HONASA.NS', 'HONDAPOWER.NS', 'HUDCO.NS', 'HPAL.NS', 'HPL.NS', 'HUHTAMAKI.NS', 'ICRA.NS',
    'IDEA.NS', 'IDFC.NS', 'IFBIND.NS', 'IFCI.NS', 'IFGLEXPOR.NS', 'IGPL.NS', 'IGARASHI.NS', 'IIFL.NS', 'IIFLSEC.NS', 'IKIO.NS',
    'IMAGICAA.NS', 'INDIACEM.NS', 'INDIAGLYCO.NS', 'INDNIPPON.NS', 'IPL.NS', 'INDIASHLTR.NS', 'ITDC.NS', 'IBULHSGFIN.NS',
    'IBREALEST.NS', 'INDIAMART.NS', 'IEX.NS', 'INDIANHUME.NS', 'IMFA.NS', 'INDIGOPNTS.NS', 'INDOAMIN.NS', 'ICIL.NS',
    'INDORAMA.NS', 'INDOCO.NS', 'INDRAMEDCO.NS', 'INFIBEAM.NS', 'INFOBEAN.NS', 'INGERRAND.NS', 'INNOVACAP.NS', 'INOXGREEN.NS',
    'INOXINDIA.NS', 'INOXWIND.NS', 'INSECTICID.NS', 'INTELLECT.NS', 'IOLCP.NS', 'IONEXCHANG.NS', 'IRB.NS', 'IRCON.NS',
    'IRMENERGY.NS', 'ISGEC.NS', 'ITDCEM.NS', 'ITI.NS', 'JKIL.NS', 'JKTYRE.NS', 'JBCHEPHARM.NS', 'JKCEMENT.NS', 'JAGRAN.NS',
    'JAGSNPHARM.NS', 'JAIBALAJI.NS', 'JAICORPLTD.NS', 'JISLJALEQS.NS', 'JPASSOCIAT.NS', 'JPPOWER.NS', 'J&KBANK.NS', 'JAMNAAUTO.NS',
    'JAYBARMARU.NS', 'JAYAGROGN.NS', 'JAYNECOIND.NS', 'JBMA.NS', 'JINDRILL.NS', 'JINDALPOLY.NS', 'JINDALSAW.NS', 'JSL.NS',
    'JINDWORLD.NS', 'JKLAKSHMI.NS', 'JKPAPER.NS', 'JMFINANCIL.NS', 'JCHAC.NS', 'JSWHL.NS', 'JTEKTINDIA.NS', 'JTLIND.NS',
    'JUBLINDS.NS', 'JUBLINGREA.NS', 'JUBLPHARMA.NS', 'JLHL.NS', 'JWL.NS', 'JUSTDIAL.NS', 'JYOTHYLAB.NS', 'JYOTIRES.BO',
    'KABRAEXTRU.NS', 'KAJARIACER.NS', 'KPIL.NS', 'KALYANKJIL.NS', 'KALYANIFRG.NS', 'KSL.NS', 'KAMAHOLD.BO', 'KAMDHENU.NS',
    'KAMOPAINTS.NS', 'KANORICHEM.NS', 'KTKBANK.NS', 'KSCL.NS', 'KAYNES.NS', 'KDDL.NS', 'KEC.NS', 'KEI.NS', 'KELLTONTEC.NS',
    'KENNAMET.NS', 'KESORAMIND.NS', 'KKCL.NS', 'RUSTOMJEE.NS', 'KFINTECH.NS', 'KHAICHEM.NS', 'KINGFA.NS', 'KIOCL.NS',
    'KIRIINDUS.NS', 'KIRLOSBROS.NS', 'KIRLFER.NS', 'KIRLOSIND.NS', 'KIRLOSENG.NS', 'KIRLPNU.NS', 'KITEX.NS', 'KMCSHIL.BO',
    'KNRCON.NS', 'KOKUYOCMLN.NS', 'KOLTEPATIL.NS', 'KOPRAN.NS', 'KOVAI.NS', 'KPIGREEN.NS', 'KPITTECH.NS', 'KPRMILL.NS',
    'KRBL.NS', 'KIMS.NS', 'KRSNAA.NS', 'KSB.NS', 'KSOLVES.NS', 'KUANTUM.NS', 'LAOPALA.NS', 'LAXMIMACH.NS', 'LANCER.NS',
    'LANDMARK.NS', 'LATENTVIEW.NS', 'LXCHEM.NS', 'LEMONTREE.NS', 'LGBBROSLTD.NS', 'LIKHITHA.NS', 'LINC.NS', 'LINCOLN.NS',
    'LLOYDSENGG.NS', 'LLOYDSENT.BO', 'LLOYDSME.NS', 'LTFOODS.NS', 'LUMAXTECH.NS', 'LUMAXIND.NS', 'MMFL.NS', 'MKPL.NS',
    'MCLOUD.BO', 'MGL.NS', 'MTNL.NS', 'MAHSCOOTER.NS', 'MAHSEAMLES.NS', 'MHRIL.NS', 'MAHLIFE.NS', 'MAHLOG.NS', 'MANINDS.NS',
    'MANINFRA.NS', 'MANGCHEFER.NS', 'MANAKSIA.NS', 'MANALIPETC.NS', 'MANAPPURAM.NS', 'MANGLMCEM.NS', 'MRPL.NS', 'MVGJL.NS',
    'MANORAMA.NS', 'MARATHON.NS', 'MARKSANS.NS', 'MASFIN.NS', 'MASTEK.NS', 'MATRIMONY.NS', 'MAXESTATES.NS', 'MAYURUNIQ.NS',
    'MAZDOCK.NS', 'MEDICAMEQ.NS', 'MEDPLUS.NS', 'MOL.NS', 'MENONBE.NS', 'METROBRAND.NS', 'METROPOLIS.NS', 'MINDACORP.NS',
    'MIRZAINT.NS', 'MIDHANI.NS', 'MISHTANN.BO', 'MMTC.NS', 'MOIL.NS', 'MOLDTKPAC.NS', 'MONARCH.NS', 'MONTECARLO.NS',
    'MOREPENLAB.NS', 'MOSCHIP.BO', 'MSUMI.NS', 'MOTILALOFS.NS', 'MPSLTD.NS', 'BECTORFOOD.NS', 'MSTCLTD.NS', 'MTARTECH.NS',
    'MUKANDLTD.NS', 'MCX.NS', 'MUTHOOTMF.NS', 'NACLIND.NS', 'NAGAFERT.NS', 'NAHARCAP.NS', 'NAHARSPING.NS', 'NSIL.NS', 'NH.NS',
    'NATCOPHARM.NS', 'NATIONALUM.NS', 'NFL.NS', 'NAVINFLUOR.NS', 'NAVKARCORP.NS', 'NAVNETEDUL.NS', 'NAZARA.NS', 'NBCC.NS',
    'NCC.NS', 'NCLIND.NS', 'NELCAST.NS', 'NELCO.NS', 'NEOGEN.NS', 'NESCO.NS', 'NETWEB.NS', 'NETWORK18.NS', 'NEULANDLAB.NS',
    'NDTV.NS', 'NEWGEN.NS', 'NGLFINE.NS', 'NIITMTS.NS', 'NIITLTD.NS', 'NIKHILAD.BO', 'NILKAMAL.NS', 'NITINSPIN.NS',
    'NITTAGELA.BO', 'NLCINDIA.NS', 'NSLNISP.NS', 'NOCIL.NS', 'NOVARTIND.NS', 'NRBBEARING.NS', 'NUCLEUS.NS', 'NUVAMA.NS',
    'OLECTRA.NS', 'OMAXE.NS', 'ONMOBILE.NS', 'ONWARDTEC.NS', 'OPTIEMUS.NS', 'ORIENTBELL.NS', 'ORIENTCEM.NS', 'ORIENTELEC.NS',
    'GREENPOWER.NS', 'ORIENTPPR.NS', 'OAL.NS', 'OCCL.NS', 'ORIENTHOT.NS', 'OSWALGREEN.NS', 'PAISALO.NS', 'PANACEABIO.NS',
    'PANAMAPET.NS', 'PARADEEP.NS', 'PARAGMILK.NS', 'PARACABLES.NS', 'PARAS.NS', 'PATELENG.NS', 'PAUSHAKLTD.BO', 'PCJEWELLER.NS',
    'PCBL.NS', 'PDSL.NS', 'PGIL.NS', 'PENIND.NS', 'PERMAGN.BO', 'PFIZER.NS', 'PGEL.NS', 'PHOENIXLTD.NS', 'PILANIINVS.NS',
    'PPLPHARMA.NS', 'PITTIENG.NS', 'PIXTRANS.NS', 'PNBGILTS.NS', 'PNBHOUSING.NS', 'PNCINFRA.NS', 'POKARNA.NS', 'POLYMED.NS',
    'POLYPLEX.NS', 'POONAWALLA.NS', 'POWERMECH.NS', 'PRAJIND.NS', 'PRAKASHSTL.NS', 'DIAMONDYD.NS', 'PRAVEG.BO', 'PRECAM.NS',
    'PRECWIRE.NS', 'PRESTIGE.NS', 'PRICOLLTD.NS', 'PFOCUS.NS', 'PRIMO.BO', 'PRINCEPIPE.NS', 'PRSMJOHNSN.NS', 'PRIVISCL.NS',
    'PGHL.NS', 'PROTEAN.BO', 'PRUDENT.NS', 'PSPPROJECT.NS', 'PFS.NS', 'PTC.NS', 'PTCIL.BO', 'PSB.NS', 'PUNJABCHEM.NS',
    'PURVA.NS', 'PVRINOX.NS', 'QUESS.NS', 'QUICKHEAL.NS', 'QUINT.BO', 'RRKABEL.NS', 'RSYSTEMS.NS', 'RACLGEAR.BO', 'RADIANTCMS.NS',
    'RADICO.NS', 'RPEL.BO', 'RVNL.NS', 'RAILTEL.NS', 'RAIN.NS', 'RAINBOW.NS', 'RAJRATAN.NS', 'RALLIS.NS', 'RAMRAT.BO',
    'RAMASTEEL.NS', 'RAMCOIND.NS', 'RAMCOSYS.NS', 'RKFORGE.NS', 'RAMKY.NS', 'RANEHOLDIN.NS', 'RML.NS', 'RCF.NS', 'RATEGAIN.NS',
    'RATNAMANI.NS', 'RTNINDIA.NS', 'RTNPOWER.NS', 'RAYMOND.NS', 'RBLBANK.NS', 'REDINGTON.NS', 'REDTAPE.NS', 'REFEX.NS',
    'RIIL.NS', 'RELINFRA.NS', 'RPOWER.NS', 'RELIGARE.NS', 'RGL.NS', 'REPCOHOME.NS', 'REPRO.NS', 'RESPONIND.NS', 'RBA.NS',
    'RHIM.NS', 'RICOAUTO.NS', 'RISHABH.NS', 'RITES.NS', 'ROLEXRINGS.NS', 'ROSSARI.NS', 'ROSSELLIND.NS', 'ROTO.NS', 'ROUTE.NS',
    'ROHLTD.NS', 'RPGLIFE.NS', 'RPSGVENT.NS', 'RSWM.NS', 'RUBYMILLS.NS', 'RUPA.NS', 'RUSHIL.NS', 'SCHAND.NS', 'SHK.NS',
    'SJS.NS', 'SPAL.NS', 'SADHNANIQ.NS', 'SAFARI.NS', 'SAGCEM.NS', 'KALAMANDIR.NS', 'SALASAR.NS', 'SAMHI.NS', 'SANDHAR.NS',
    'SANDUMA.NS', 'SANGAMIND.NS', 'SANGHIIND.NS', 'SANGHVIMOV.NS', 'SANINFRA.BO', 'SANOFI.NS', 'SANSERA.NS', 'SAPPHIRE.NS',
    'SARDAEN.NS', 'SAREGAMA.NS', 'SASKEN.NS', 'SASTASUNDR.NS', 'SATINDLTD.NS', 'SATIA.NS', 'SATIN.NS', 'SOTL.NS', 'SBFC.NS',
    'SCHNEIDER.NS', 'SEAMECLTD.NS', 'SENCO.NS', 'SEPC.NS', 'SEQUENT.NS', 'SESHAPAPER.NS', 'SGFIN.BO', 'SHAILY.NS',
    'SHAKTIPUMP.NS', 'SHALBY.NS', 'SHALPAINTS.NS', 'SHANKARA.NS', 'SHANTIGEAR.NS', 'SHARDACROP.NS', 'SHARDAMOTR.NS',
    'SHAREINDIA.NS', 'SFL.NS', 'SHILPAMED.NS', 'SCI.NS', 'SHIVACEM.BO', 'SBCL.NS', 'SHIVALIK.NS', 'SHOPERSTOP.NS',
    'SHREDIGCEM.NS', 'SHREEPUSHK.NS', 'RENUKA.NS', 'SHREYAS.NS', 'SHRIRAMPPS.NS', 'SHYAMMETL.NS', 'SIGACHI.NS', 'SIGNATURE.NS',
    'SIRCA.NS', 'SIS.NS', 'SIYSIL.NS', 'SKFINDIA.NS', 'SKIPPER.NS', 'SMCGLOBAL.NS', 'SMLISUZU.NS', 'SMSPHARMA.NS',
    'SNOWMAN.NS', 'SOBHA.NS', 'SOLARA.NS', 'SDBL.NS', 'SOMANYCERA.NS', 'SONATSOFTW.NS', 'SOUTHBANK.NS', 'SPANDANA.NS',
    'SPECIALITY.NS', 'SPENCERS.NS', 'SPICEJET.NS', 'SPORTKING.NS', 'SRHHYPOLTD.NS', 'SAINTGOBAIN.BO', 'STARCEMENT.NS',
    'STEELXIND.NS', 'SSWL.NS', 'STEELCAS.NS', 'SWSOLAR.NS', 'STERTOOLS.NS', 'STLTECH.NS', 'STOVEKRAFT.NS', 'STAR.NS',
    'STYLAMIND.NS', 'STYRENIX.NS', 'SUBEXLTD.NS', 'SUBROS.NS', 'SUDARSCHEM.NS', 'SUKHJITS.NS', 'SULA.NS', 'SUMICHEM.NS',
    'SUMMITSEC.NS', 'SPARC.NS', 'SUNCLAY.NS', 'SUNDRMFAST.NS', 'SUNFLAG.NS', 'SUNTECK.NS', 'SUPRAJIT.NS', 'SPLPETRO.NS',
    'SUPRIYA.NS', 'SURAJEST.NS', 'SURYAROSNI.NS', 'SURYODAY.NS', 'SUTLEJTEX.NS', 'SUVEN.NS', 'SUVENPHAR.NS', 'SUZLON.NS',
    'SWANENERGY.NS', 'SWARAJENG.NS', 'SYMPHONY.NS', 'SYNCOMF.NS', 'SYNGENE.NS', 'SYRMA.NS', 'TAJGVK.NS', 'TALBROAUTO.NS',
    'TMB.NS', 'TNPL.NS', 'TNPETRO.NS', 'TANFACIND.BO', 'TANLA.NS', 'TARC.NS', 'TARSONS.NS', 'TASTYBITE.NS', 'TATACHEM.NS',
    'TATAINVEST.NS', 'TTML.NS', 'TATVA.NS', 'TCIEXP.NS', 'TCNSBRANDS.NS', 'TCPLPACK.NS', 'TDPOWERSYS.NS', 'TEAMLEASE.NS',
    'TECHNOE.NS', 'TIIL.NS', 'TEGA.NS', 'TEJASNET.NS', 'TEXRAIL.NS', 'TGVSL.BO', 'THANGAMAYL.NS', 'ANUP.NS', 'THEMISMED.NS',
    'THERMAX.NS', 'TIRUMALCHM.NS', 'THOMASCOOK.NS', 'THYROCARE.NS', 'TI.NS', 'TIMETECHNO.NS', 'TIMEX.BO', 'TIMKEN.NS',
    'TIPSINDLTD.NS', 'TITAGARH.NS', 'TFCILTD.NS', 'TRACXN.NS', 'TRIL.NS', 'TREL.NS', 'TRANSPEK.BO', 'TCI.NS', 'TRIDENT.NS',
    'TRIVENI.NS', 'TRITURBINE.NS', 'TRU.NS', 'TTKHLTCARE.NS', 'TTKPRESTIG.NS', 'TVTODAY.NS', 'TV18BRDCST.NS', 'TVSELECT.NS',
    'TVSHLTD.NS', 'TVSSRICHAK.NS', 'TVSSCS.NS', 'UDAICEMENT.NS', 'UFLEX.NS', 'UGARSUGAR.NS', 'UGROCAP.NS', 'UJJIVANSFB.NS',
    'ULTRAMAR.BO', 'UNICHEMLAB.NS', 'UNIPARTS.NS', 'UNIVCABLES.NS', 'UDS.NS', 'USHAMART.NS', 'UTIAMC.NS', 'UTKARSHBNK.NS',
    'UTTAMSUGAR.NS', 'VGUARD.NS', 'VMART.NS', 'VSTTILLERS.NS', 'WABAG.NS', 'VADILALIND.NS', 'VAIBHAVGBL.NS', 'VAKRANGEE.NS',
    'VALIANTORG.NS', 'DBREALTY.NS', 'VSSL.NS', 'VTL.NS', 'VARROC.NS', 'VASCONEQ.NS', 'VENKEYS.NS', 'VENUSPIPES.NS',
    'VERANDA.NS', 'VESUVIUS.NS', 'VIDHIING.NS', 'VIJAYA.NS', 'VIKASLIFE.NS', 'VIMTALABS.NS', 'VINATIORGA.NS', 'VINDHYATEL.NS',
    'VINYLINDIA.NS', 'VIPIND.NS', 'VISAKAIND.NS', 'VISHNU.NS', 'VPRPL.NS', 'VLSFINANCE.NS', 'VOLTAMP.NS', 'VRLLOG.NS',
    'VSTIND.NS', 'WAAREERTL.BO', 'WARDINMOBI.BO', 'WELCORP.NS', 'WELENT.NS', 'WELSPUNLIV.NS', 'WELSPLSOL.NS', 'WENDT.NS',
    'WSTCSTPAPR.NS', 'WESTLIFE.NS', 'WOCKPHARMA.NS', 'WONDERLA.NS', 'WPIL.BO', 'XCHANGING.NS', 'YASHO.NS', 'YATHARTH.NS',
    'YATRA.NS', 'YUKEN.NS', 'ZAGGLE.NS', 'ZEEMEDIA.NS', 'ZENTEC.NS', 'ZENSARTECH.NS', 'ZFCVINDIA.NS', 'ZUARI.NS', 'ZYDUSWELL.NS'
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
st.set_page_config(layout="wide", page_title="TradeSense", page_icon="📈",initial_sidebar_state="expanded")

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
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            st.error("Email already exists. Try a different email.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            hashed_password = hash_password(password)
            new_user = {
                "name": name,
                "email": email,
                "password": hashed_password,
                "dob": dob.strftime('%Y-%m-%d'), 
                "pob": pob
            }
            users_collection.insert_one(new_user)
            st.success("User registered successfully!")

# Login function
def login():
    st.subheader("Login")
    email = st.text_input("Enter your email", key='login_email')
    password = st.text_input("Enter your password", type="password", key='login_password')

    if st.button("Login"):
        user = users_collection.find_one({"email": email})
        if user and verify_password(user['password'], password):
            st.success("Login successful!")
            st.session_state.logged_in = True
            st.session_state.username = user['name']
            st.session_state.email = user['email']
            st.session_state.user_id = user['_id']
        else:
            st.error("Invalid email or password.")

# Forgot password function
def forgot_password():
    st.subheader("Forgot Password")
    email = st.text_input("Enter your email", key='forgot_email')
    dob = st.date_input("Enter your date of birth", key='forgot_dob')
    pob = st.text_input("Enter your place of birth", key='forgot_pob')

    if st.button("Submit"):
        user = users_collection.find_one({"email": email, "dob": dob, "pob": pob})
        if user:
            st.session_state.email = email
            st.session_state.user_id = user['_id']
            st.session_state.identity_verified = True
            st.success("Identity verified. Please reset your password.")
        else:
            st.error("Invalid details provided.")

    if st.session_state.identity_verified:
        new_password = st.text_input("Enter a new password", type="password", key='reset_new_password')
        confirm_new_password = st.text_input("Confirm your new password", type="password", key='reset_confirm_new_password')

        if st.button("Reset Password"):
            if new_password != confirm_new_password:
                st.error("Passwords do not match.")
            else:
                hashed_password = hash_password(new_password)
                users_collection.update_one({"_id": ObjectId(st.session_state.user_id)}, {"$set": {"password": hashed_password}})
                st.success("Password reset successfully. You can now log in with the new password.")
                st.session_state.identity_verified = False

# Function to convert string date to datetime.date
def string_to_date(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d').date()

# My Account function
def my_account():
    st.subheader("My Account")

    if st.session_state.logged_in:
        user = users_collection.find_one({"_id": ObjectId(st.session_state.user_id)})

        new_name = st.text_input("Update your name", value=user['name'], key='account_name')
        new_dob = st.date_input("Update your date of birth", value=string_to_date(user['dob']), key='account_dob')
        new_pob = st.text_input("Update your place of birth", value=user['pob'], key='account_pob')

        if st.button("Update Details"):
            users_collection.update_one({"_id": ObjectId(user['_id'])}, {"$set": {"name": new_name, "dob": new_dob, "pob": new_pob}})
            st.success("Details updated successfully!")

        st.subheader("Change Password")
        current_password = st.text_input("Enter your current password", type="password", key='account_current_password')
        new_password = st.text_input("Enter a new password", type="password", key='account_new_password')
        confirm_new_password = st.text_input("Confirm your new password", type="password", key='account_confirm_new_password')

        if st.button("Change Password"):
            if verify_password(user['password'], current_password):
                if new_password != confirm_new_password:
                    st.error("Passwords do not match.")
                else:
                    hashed_password = hash_password(new_password)
                    users_collection.update_one({"_id": ObjectId(user['_id'])}, {"$set": {"password": hashed_password}})
                    st.success("Password changed successfully!")
            else:
                st.error("Current password is incorrect.")

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.email = ""
    st.session_state.user_id = None

# Display tables function
def display_tables():
    st.title('Database Admin Page')

    st.header("Users Collection")
    users = list(users_collection.find())
    st.dataframe(pd.DataFrame(users))

    st.header("Watchlists Collection")
    watchlists = list(watchlists_collection.find())
    st.dataframe(pd.DataFrame(watchlists))

    st.header("Portfolios Collection")
    portfolios = list(portfolios_collection.find())
    st.dataframe(pd.DataFrame(portfolios))

# Helper function to fetch ticker data from yfinance
def fetch_ticker_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        if data.empty:
            raise ValueError("Ticker not found")
        return data
    except Exception as e:
        raise ValueError("Ticker not found") from e

# Helper function to calculate indicators
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

def get_user_id(email):
    user = users_collection.find_one({"email": email})
    return user['_id'] if user else None


# Watchlist feature
def display_watchlist():
    st.header(f"{st.session_state.username}'s Watchlist")
    user_id = get_user_id(st.session_state.email)
    watchlist = list(watchlists_collection.find({"user_id": user_id}))

    # Add new ticker to watchlist
    new_ticker = st.text_input("Add a new ticker to your watchlist")
    if st.button("Add Ticker"):
        try:
            fetch_ticker_data(new_ticker)
            if not watchlists_collection.find_one({"user_id": user_id, "ticker": new_ticker}):
                watchlists_collection.insert_one({"user_id": user_id, "ticker": new_ticker})
                st.success(f"{new_ticker} added to your watchlist!")
            else:
                st.warning(f"{new_ticker} is already in your watchlist.")
        except ValueError as ve:
            st.error(ve)

    # Display watchlist
    if watchlist:
        watchlist_data = {}
        for entry in watchlist:
            ticker = entry['ticker']
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
        ticker_to_remove = st.selectbox("Select a ticker to remove", [entry['ticker'] for entry in watchlist])
        if st.button("Remove Ticker"):
            watchlists_collection.delete_one({"user_id": user_id, "ticker": ticker_to_remove})
            st.success(f"{ticker_to_remove} removed from your watchlist.")
            st.experimental_rerun()  # Refresh the app to reflect changes
    else:
        st.write("Your watchlist is empty.")

# Portfolio feature
def display_portfolio():
    st.header(f"{st.session_state.username}'s Portfolio")
    user_id = get_user_id(st.session_state.email)
    portfolio = list(portfolios_collection.find({"user_id": user_id}))

    # Add new stock to portfolio
    st.subheader("Add to Portfolio")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_ticker = st.text_input("Ticker Symbol")
    with col2:
        shares = st.number_input("Number of Shares", min_value=0.0, step=0.01)
    with col3:
        bought_price = st.number_input("Bought Price per Share", min_value=0.0, step=0.01)
    if st.button("Add to Portfolio"):
        try:
            current_data = yf.download(new_ticker, period='1d')
            if current_data.empty:
                raise ValueError("Ticker not found")

            if not portfolios_collection.find_one({"user_id": user_id, "ticker": new_ticker}):
                portfolios_collection.insert_one({
                    "user_id": user_id,
                    "ticker": new_ticker,
                    "shares": shares,
                    "bought_price": bought_price,
                    "date_added": pd.Timestamp.now()
                })
                st.success(f"{new_ticker} added to your portfolio!")
            else:
                st.warning(f"{new_ticker} is already in your portfolio.")
        except Exception as e:
            st.error(f"Error adding ticker: {e}")

    # Display portfolio
    if portfolio:
        portfolio_data = []
        invested_values = []
        current_values = []
        for entry in portfolio:
            try:
                current_data = yf.download(entry['ticker'], period='1d')
                if current_data.empty:
                    raise ValueError(f"Ticker {entry['ticker']} not found")

                last_price = current_data['Close'].iloc[-1]
                invested_value = entry['shares'] * entry['bought_price']
                current_value = entry['shares'] * last_price
                p_l = current_value - invested_value
                p_l_percent = (p_l / invested_value) * 100
                portfolio_data.append({
                    "Ticker": entry['ticker'],
                    "Shares": entry['shares'],
                    "Bought Price": entry['bought_price'],
                    "Invested Value": invested_value,
                    "Last Traded Price": last_price,
                    "Current Value": current_value,
                    "P&L (%)": p_l_percent
                })
                invested_values.append(invested_value)
                current_values.append(current_value)
            except Exception as e:
                st.error(f"Error retrieving data for {entry['ticker']}: {e}")

        portfolio_df = pd.DataFrame(portfolio_data)

        st.write("Your Portfolio:")
        st.dataframe(portfolio_df)

        col1, col2 = st.columns(2)

        with col1:
            labels = portfolio_df['Ticker']
            values = portfolio_df['Current Value']
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig.update_layout(title_text="Portfolio Distribution")
            st.plotly_chart(fig)

        with col2:
            total_invested_value = sum(invested_values)
            total_current_value = sum(current_values)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=['Total Invested Value', 'Total Current Value'], y=[total_invested_value, total_current_value]))
            fig.update_layout(title_text='Profit/Loss', xaxis_title='Value', yaxis_title='Sum')
            st.plotly_chart(fig)

        ticker_to_remove = st.selectbox("Select a ticker to remove", [entry['ticker'] for entry in portfolio])
        if st.button("Remove from Portfolio"):
            portfolios_collection.delete_one({"user_id": user_id, "ticker": ticker_to_remove})
            st.success(f"{ticker_to_remove} removed from your portfolio.")
            st.experimental_rerun()  # Refresh the app to reflect changes
    else:
        st.write("Your portfolio is empty.")

# Main menu function
def main_menu():
    st.subheader("Main Menu")
    menu_options = [f"{st.session_state.username}'s Portfolio",f"{st.session_state.username}'s Watchlist", "Stock Screener", "Stock Analysis",
                    "Markets", "My Account", "Database Admin Page"]
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

######################################################### Main content area ######################################################################

if not st.session_state.logged_in:
    # dashboard code-------------------------------------------------------------------------------------------------------

    st.title("TradeSense")
    st.write("An ultimate platform for smart trading insights. Please log in or sign up to get started.")

    st.subheader("Market Performance")

    # Function to get stock data and calculate moving averages
    @st.cache_data
    def get_stock_data(ticker_symbol, start_date, end_date):
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data.dropna(inplace=True)
        return data

    # Function to create Plotly figure with volume histogram
    def create_figure(data, indicators, title):
        fig = go.Figure()

        # Add candlestick chart
        fig.add_trace(go.Candlestick(x=data.index,
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close'],
                                    name='Candlesticks'))

        if 'Close' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        if 'MA_10' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_10'], mode='lines', name='10-day MA'))
        if 'MA_20' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], mode='lines', name='20-day MA'))

        # Add volume histogram
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', yaxis='y2', marker_color='rgba(0, 0, 100, 0.5)'))

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
            yaxis=dict(
                title='Price',
                fixedrange=False
            ),
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Reset Zoom",
                            method="relayout",
                            args=[{"xaxis.range": [None, None],
                                    "yaxis.range": [None, None]}])]
            )]
        )
        return fig

    col1, col2, col3 = st.columns(3)

    with col1:
        stock_symbols = {
            "BSE 500": "BSE-500.BO",
            "NIFTY 50": "^NSEI",
            "S&P 500": "^GSPC",
            "FTSE 100": "^FTSE",
            "SSE Composite (China)": "000001.SS",
            "Nikkei 225 (Japan)": "^N225",
            "ASX 200 (Australia)": "^AXJO",
            "S&P/TSX (Canada)": "^GSPTSE",
            "Bitcoin": "BTC-USD",
            "EUR/USD": "EURUSD=X",
            "Gold Futures": "GC=F",
            "Crude Oil Futures": "CL=F"
        }
        stock_name = st.selectbox("Select Stock", list(stock_symbols.keys()))
        ticker = stock_symbols[stock_name]
        st.write(f"You selected: {stock_name}")

    with col2:
        START = st.date_input('Start Date', value=datetime.now() - timedelta(days=365))

    with col3:
        END = st.date_input('End Date', value=datetime.now() + timedelta(days=1))

    if ticker and START and END:
        data = get_stock_data(ticker, START, END)
        fig = create_figure(data, ['Close', 'MA_10', 'MA_20'], f"{stock_name} Stock Prices")
        st.plotly_chart(fig)

    # Market Performance
    st.subheader("Market's Performance")

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

    # Define date ranges
    date_ranges = {
        "1 day": timedelta(days=1),
        "5 days": timedelta(days=5),
        "10 days": timedelta(days=10),
        "1 month": timedelta(days=30),
        "3 months": timedelta(days=90),
        "6 months": timedelta(days=180),
        "1 year": timedelta(days=365),
        "2 years": timedelta(days=730),
        "3 years": timedelta(days=1095),
        "5 years": timedelta(days=1825)
    }

    selected_range = st.select_slider(
        "Select Date Range for Market Performance",
        options=list(date_ranges.keys()),
        value="1 year"
    )
    END = datetime.now()
    START = END - date_ranges[selected_range]

    def get_market_data(ticker_symbol, start_date, end_date):
        return yf.download(ticker_symbol, start=start_date, end=end_date)

    def calculate_performance(data):
        if data is not None and not data.empty:
            performance = (data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100
            return performance
        return None

    market_performance = {
        market: calculate_performance(get_market_data(ticker, START, END))
        for market, ticker in market_indices.items()
        if calculate_performance(get_market_data(ticker, START, END)) is not None
    }

    performance_df = pd.DataFrame(list(market_performance.items()), columns=['Market', 'Performance'])
    fig = px.bar(performance_df, x='Market', y='Performance', title='Market Performance',
                labels={'Performance': 'Performance (%)'}, color='Performance',
                color_continuous_scale=px.colors.diverging.RdYlGn)
    st.plotly_chart(fig)

else:
    if choice:
        if choice == "My Account":
            my_account()
        elif choice == f"{st.session_state.username}'s Watchlist":
            display_watchlist()

        elif choice == f"{st.session_state.username}'s Portfolio":
            display_portfolio()

        elif choice == "Markets":
                #'Markets' code-------------------------------------------------------------------------------------------------------------------------------------------------------

                # Function to download data and calculate moving averages with caching
                @st.cache_data
                def get_stock_data(ticker_symbol):
                    data = yf.download(ticker_symbol, period='1y')
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
                                    xaxis_rangeslider_visible=True, hovermode='x',
                                    xaxis=dict(rangeselector=dict(buttons=list([
                                        dict(count=1, label="1m", step="month", stepmode="backward"),
                                        dict(count=6, label="6m", step="month", stepmode="backward"),
                                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                                        dict(count=1, label="1y", step="year", stepmode="backward"),
                                        dict(step="all")
                                    ])), rangeslider=dict(visible=True), type='date'),
                                    yaxis=dict(fixedrange=False),
                                    updatemenus=[dict(type="buttons", buttons=[dict(label="Reset Zoom",
                                                                                    method="relayout",
                                                                                    args=[{"xaxis.range": [None, None],
                                                                                            "yaxis.range": [None, None]}])])])
                    return fig

                # Function to calculate correlation
                def calculate_correlation(data1, data2):
                    return data1['Close'].corr(data2['Close'])

                # Function to plot correlation matrix
                def plot_correlation_matrix(correlation_matrix):
                    fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
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
                    aligned_returns = pd.concat([asset_returns, market_returns], axis=1).dropna()
                    covariance_matrix = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])
                    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
                    return beta

                # Function to calculate Value at Risk (VaR)
                def calculate_var(returns, confidence_level=0.05):
                    return np.percentile(returns, confidence_level * 100)

                # Main application

                # Markets submenu
                submenu = st.sidebar.selectbox("Select Option", ["Equities", "Commodities", "Currencies", "Cryptocurrencies", "Insights"])

                if submenu == "Equities":
                    ticker_category = st.sidebar.selectbox("Select Index", ["BSE-LargeCap", "BSE-MidCap", "BSE-SmallCap"])
                    tickers = {"BSE-LargeCap": bse_largecap, "BSE-MidCap": bse_midcap, "BSE-SmallCap": bse_smallcap}[ticker_category]

                    @st.cache_data
                    def get_sector_industry_price_changes(tickers):
                        data = {
                            'Ticker': [], 'Company Name': [], 'Sector': [], 'Industry': [], 'Market Cap': [], 'Last Traded Price': [],
                            '1D % Change': [], '2D % Change': [], '3D % Change': [], '5D % Change': [], '2W % Change': [],
                            '1M % Change': [], '3M % Change': [], '6M % Change': [], '1Y % Change': [],
                            '1D Volume': [], '2D Volume': [], '5D Volume': [], '2W Volume': [],
                            '1M Volume': [], '3M Volume': [], '6M Volume': [], '1Y Volume': [],
                            'Volume Change %': []
                        }
                        for ticker in tickers:
                            try:
                                stock = yf.Ticker(ticker)
                                info = stock.info
                                price_data_1y = yf.download(ticker, period='1y')
                                
                                if not price_data_1y.empty:
                                    last_traded_price = price_data_1y['Close'].iloc[-1]
                                    one_day_volume = price_data_1y['Volume'].iloc[-1]
                                    two_day_volume = price_data_1y['Volume'].iloc[-2:].mean()
                                    five_day_volume = price_data_1y['Volume'].iloc[-5:].mean()
                                    two_week_volume = price_data_1y['Volume'].iloc[-10:].mean()
                                    one_month_volume = price_data_1y['Volume'].iloc[-21:].mean()
                                    three_month_volume = price_data_1y['Volume'].iloc[-63:].mean()
                                    six_month_volume = price_data_1y['Volume'].iloc[-126:].mean()
                                    one_year_volume = price_data_1y['Volume'].mean()
                                    avg_volume = price_data_1y['Volume'].mean()
                                    volume_change = ((one_day_volume - avg_volume) / avg_volume) * 100 if avg_volume != 0 else 'N/A'
                                    price_changes = price_data_1y['Close'].pct_change() * 100
                                    one_day_change = price_changes.iloc[-1]
                                    two_day_change = price_changes.iloc[-2:].sum()
                                    three_day_change = price_changes.iloc[-3:].sum()
                                    five_day_change = price_changes.iloc[-5:].sum()
                                    two_week_change = price_changes.iloc[-10:].sum()
                                    one_month_change = price_changes.iloc[-21:].sum()
                                    three_month_change = price_changes.iloc[-63:].sum()
                                    six_month_change = price_changes.iloc[-126:].sum()
                                    one_year_change = price_changes.sum()
                                else:
                                    last_traded_price = 'N/A'
                                    one_day_volume = 'N/A'
                                    two_day_volume = 'N/A'
                                    five_day_volume = 'N/A'
                                    two_week_volume = 'N/A'
                                    one_month_volume = 'N/A'
                                    three_month_volume = 'N/A'
                                    six_month_volume = 'N/A'
                                    one_year_volume = 'N/A'
                                    volume_change = 'N/A'
                                    one_day_change = 'N/A'
                                    two_day_change = 'N/A'
                                    three_day_change = 'N/A'
                                    five_day_change = 'N/A'
                                    two_week_change = 'N/A'
                                    one_month_change = 'N/A'
                                    three_month_change = 'N/A'
                                    six_month_change = 'N/A'
                                    one_year_change = 'N/A'
                                
                                data['Ticker'].append(ticker)
                                data['Company Name'].append(info.get('longName', 'N/A'))
                                data['Sector'].append(info.get('sector', 'N/A'))
                                data['Industry'].append(info.get('industry', 'N/A'))
                                data['Last Traded Price'].append(last_traded_price)
                                data['Market Cap'].append(info.get('marketCap', 'N/A'))
                                data['1D % Change'].append(one_day_change)
                                data['2D % Change'].append(two_day_change)
                                data['3D % Change'].append(three_day_change)
                                data['5D % Change'].append(five_day_change)
                                data['2W % Change'].append(two_week_change)
                                data['1M % Change'].append(one_month_change)
                                data['3M % Change'].append(three_month_change)
                                data['6M % Change'].append(six_month_change)
                                data['1Y % Change'].append(one_year_change)
                                data['1D Volume'].append(one_day_volume)
                                data['2D Volume'].append(two_day_volume)
                                data['5D Volume'].append(five_day_volume)
                                data['2W Volume'].append(two_week_volume)
                                data['1M Volume'].append(one_month_volume)
                                data['3M Volume'].append(three_month_volume)
                                data['6M Volume'].append(six_month_volume)
                                data['1Y Volume'].append(one_year_volume)
                                data['Volume Change %'].append(volume_change)
                            
                            except Exception as e:
                                st.error(f"Error fetching data for {ticker}: {e}")
                        
                        df = pd.DataFrame(data)
                        
                        # Convert all relevant columns to numeric and fill NaNs with 0
                        numeric_columns = [col for col in df.columns if col not in ['Ticker', 'Company Name', 'Sector', 'Industry']]
                        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

                        return df

                    sector_industry_price_changes_df = get_sector_industry_price_changes(tickers)

                    # Streamlit app
                    st.subheader('Market Stats')
                    st.dataframe(sector_industry_price_changes_df)
                    st.subheader('Price')
                    price_chart_option = st.selectbox('Select period to view price changes:', [
                        '1D % Change', '2D % Change', '3D % Change', '5D % Change',
                        '2W % Change', '1M % Change', '3M % Change', '6M % Change', '1Y % Change'
                    ])

                    df_price_sorted = sector_industry_price_changes_df[['Ticker', price_chart_option]].copy()
                    df_price_sorted[price_chart_option] = pd.to_numeric(df_price_sorted[price_chart_option], errors='coerce')
                    df_price_sorted = df_price_sorted.sort_values(by=price_chart_option, ascending=False).reset_index(drop=True)
                    df_price_sorted.columns = ['Ticker', '% Change']

                    fig_price = px.bar(df_price_sorted, x='Ticker', y='% Change', title=f'{price_chart_option} Gainers/Losers', color='% Change', color_continuous_scale=px.colors.diverging.RdYlGn)
                    st.plotly_chart(fig_price)

                    st.subheader('Volume')
                    volume_chart_option = st.selectbox('Select period to view volume changes:', [
                        '1D Volume', '2D Volume', '5D Volume', '2W Volume',
                        '1M Volume', '3M Volume', '6M Volume', '1Y Volume'
                    ])

                    df_volume_sorted = sector_industry_price_changes_df[['Ticker', volume_chart_option]].copy()
                    df_volume_sorted[volume_chart_option] = pd.to_numeric(df_volume_sorted[volume_chart_option], errors='coerce')
                    df_volume_sorted = df_volume_sorted.sort_values(by=volume_chart_option, ascending=False).reset_index(drop=True)
                    df_volume_sorted.columns = ['Ticker', 'Volume']

                    fig_volume = px.bar(df_volume_sorted, x='Ticker', y='Volume', title=f'{volume_chart_option} Volume', color='Volume', color_continuous_scale=px.colors.diverging.RdYlGn)
                    st.plotly_chart(fig_volume)

                    st.subheader('Sector and Industry Performance')
                    numeric_columns = [col for col in sector_industry_price_changes_df.columns if col not in ['Ticker', 'Company Name', 'Sector', 'Industry']]

                    sector_performance = sector_industry_price_changes_df.groupby('Sector')[numeric_columns].mean().reset_index()
                    sector_chart_option = st.selectbox('Select period to view sector performance:', numeric_columns)

                    sector_sorted = sector_performance[['Sector', sector_chart_option]].copy()
                    sector_sorted = sector_sorted.sort_values(by=sector_chart_option, ascending=False).reset_index(drop=True)
                    sector_sorted.columns = ['Sector', '% Change']

                    fig_sector = px.bar(sector_sorted, x='Sector', y='% Change', title=f'{sector_chart_option} by Sector', color='% Change', color_continuous_scale=px.colors.diverging.RdYlGn)
                    st.plotly_chart(fig_sector)

                    industry_performance = sector_industry_price_changes_df.groupby('Industry')[numeric_columns].mean().reset_index()
                    industry_chart_option = st.selectbox('Select period to view industry performance:', numeric_columns)

                    industry_sorted = industry_performance[['Industry', industry_chart_option]].copy()
                    industry_sorted = industry_sorted.sort_values(by=industry_chart_option, ascending=False).reset_index(drop=True)
                    industry_sorted.columns = ['Industry', '% Change']

                    fig_industry = px.bar(industry_sorted, x='Industry', y='% Change', title=f'{industry_chart_option} by Industry', color='% Change', color_continuous_scale=px.colors.diverging.RdYlGn)
                    st.plotly_chart(fig_industry)

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
                            data = get_stock_data(ticker)
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
                            data = get_stock_data(ticker)
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
                            data = get_stock_data(ticker)
                            fig = create_figure(data, indicators, f'{ticker} Price')
                            col.plotly_chart(fig)

                elif submenu == "Insights":
                    st.subheader("Detailed Market Analysis")
                    st.write("This section provides an in-depth analysis of the markets, commodities, forex, and cryptos.")

                    # Get data for all categories
                    data_nyse = get_stock_data("^NYA")
                    data_bse = get_stock_data("^BSESN")
                    data_gold = get_stock_data("GC=F")
                    data_oil = get_stock_data("CL=F")
                    data_eurusd = get_stock_data("EURUSD=X")
                    data_gbpusd = get_stock_data("GBPUSD=X")
                    data_btc = get_stock_data("BTC-USD")
                    data_eth = get_stock_data("ETH-USD")

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
                    data_sp500 = get_stock_data("^GSPC")
                    market_returns = data_sp500['Close'].pct_change().dropna()

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
   

            st.sidebar.subheader("Stock Screener")

            # Dropdown for selecting ticker category
            ticker_category = st.sidebar.selectbox("Select Index", ["BSE-LargeCap", "BSE-MidCap", "BSE-SmallCap"])

            # Dropdown for Strategies
            submenu = st.sidebar.selectbox("Select Strategy", ["MACD", "Moving Average", "Bollinger Bands", "Volume"])

            # Date inputs
            start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            end_date = st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=1))

            # Set tickers based on selected category
            if ticker_category == "BSE-LargeCap":
                tickers = bse_largecap
            elif ticker_category == "BSE-MidCap":
                tickers = bse_midcap
            else:
                tickers = bse_smallcap

            # Define functions for strategy logic
            def calculate_macd(data):
                macd = ta.macd(data['Close'])
                data['MACD'] = macd['MACD_12_26_9']
                data['MACD_signal'] = macd['MACDs_12_26_9']
                data['MACD_histogram'] = macd['MACDh_12_26_9']
                return data

            def check_macd_signal(data):
                recent_data = data[-5:]
                for i in range(1, len(recent_data)):
                    if (recent_data['MACD'].iloc[i] > recent_data['MACD_signal'].iloc[i] and
                        recent_data['MACD'].iloc[i-1] < recent_data['MACD_signal'].iloc[i-1] and
                        recent_data['MACD'].iloc[i] > 0 and
                        recent_data['MACD_histogram'].iloc[i] > 0 and
                        recent_data['MACD_histogram'].iloc[i-1] < 0 and
                        recent_data['MACD_histogram'].iloc[i] > recent_data['MACD_histogram'].iloc[i-1] > recent_data['MACD_histogram'].iloc[i-2]):
                        return recent_data.index[i]
                return None

            def check_bollinger_low_cross(data):
                recent_data = data[-5:]
                for i in range(1, len(recent_data)):
                    if (recent_data['Close'].iloc[i] < recent_data['BB_Low'].iloc[i] and
                        recent_data['Close'].iloc[i-1] >= recent_data['BB_Low'].iloc[i-1]):
                        return recent_data.index[i]
                return None

            def calculate_ema(data):
                data['Short_EMA'] = ta.ema(data['Close'], length=10)
                data['Long_EMA'] = ta.ema(data['Close'], length=20)
                return data

            def check_moving_average_crossover(data):
                recent_data = data[-5:]
                for i in range(1, len(recent_data)):
                    if (recent_data['Short_EMA'].iloc[i] > recent_data['Long_EMA'].iloc[i] and
                        recent_data['Short_EMA'].iloc[i-1] <= recent_data['Long_EMA'].iloc[i-1]):
                        return recent_data.index[i]
                return None

            def calculate_volume(data):
                data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
                return data

            def check_volume_increase(data):
                recent_data = data[-5:]
                for i in range(1, len(recent_data)):
                    if (recent_data['Volume'].iloc[i] > recent_data['Volume_MA'].iloc[i] and
                        recent_data['Volume'].iloc[i-1] <= recent_data['Volume_MA'].iloc[i-1]):
                        return recent_data.index[i]
                return None

            @st.cache_data
            def get_stock_data(ticker_symbols, start_date, end_date):
                try:
                    stock_data = {}
                    progress_bar = st.progress(0)
                    for idx, ticker_symbol in enumerate(ticker_symbols):
                        df = yf.download(ticker_symbol, start=start_date, end=end_date)
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

            @st.cache_data
            def calculate_indicators(df):
                df['20_MA'] = ta.sma(df['Close'], length=20)
                df['50_MA'] = ta.sma(df['Close'], length=50)

                macd = ta.macd(df['Close'])
                df['MACD'] = macd['MACD_12_26_9']
                df['MACD_Signal'] = macd['MACDs_12_26_9']
                df['MACD_Histogram'] = macd['MACDh_12_26_9']

                df['RSI'] = ta.rsi(df['Close'])

                df['Std_Dev'] = df['Close'].rolling(window=20).std()
                df['BB_High'] = df['20_MA'] + (df['Std_Dev'] * 2)
                df['BB_Low'] = df['20_MA'] - (df['Std_Dev'] * 2)

                return df

            def fetch_latest_data(tickers_with_dates):
                technical_data = []
                for ticker, occurrence_date in tickers_with_dates:
                    data = yf.download(ticker, start=start_date, end=end_date)
                    if data.empty:
                        continue
                    data['5_day_EMA'] = ta.ema(data['Close'], length=5)
                    data['10_day_EMA'] = ta.ema(data['Close'], length=10)
                    data['20_day_EMA'] = ta.ema(data['Close'], length=20)
                    macd = ta.macd(data['Close'])
                    data['MACD'] = macd['MACD_12_26_9']
                    data['MACD_Hist'] = macd['MACDh_12_26_9']
                    data['RSI'] = ta.rsi(data['Close'])
                    adx = ta.adx(data['High'], data['Low'], data['Close'])
                    data['ADX_14'] = adx['ADX_14']
                    data['ADX_NEG'] = adx['DMP_14']
                    data['ADX_POS'] = adx['DMN_14']
                    data['20_MA'] = ta.sma(data['Close'], length=20)
                    data['Std_Dev'] = data['Close'].rolling(window=20).std()
                    data['BB_High'] = data['20_MA'] + (data['Std_Dev'] * 2)
                    data['BB_Low'] = data['20_MA'] - (data['Std_Dev'] * 2)
                    data['20_day_vol_MA'] = data['Volume'].rolling(window=20).mean()

                    latest_data = data.iloc[-1]
                    technical_data.append({
                        'Ticker': ticker,
                        'Date of Occurrence': occurrence_date,
                        'Close': latest_data['Close'],
                        '5_day_EMA': latest_data['5_day_EMA'],
                        '10_day_EMA': latest_data['10_day_EMA'],
                        '20_day_EMA': latest_data['20_day_EMA'],
                        'MACD': latest_data['MACD'],
                        'MACD_Hist': latest_data['MACD_Hist'],
                        'RSI': latest_data['RSI'],
                        'ADX_14': latest_data['ADX_14'],
                        'ADX_NEG': latest_data['ADX_NEG'],
                        'ADX_POS': latest_data['ADX_POS'],
                        'Bollinger_High': latest_data['BB_High'],
                        'Bollinger_Low': latest_data['BB_Low'],
                        'Volume': latest_data['Volume'],
                        '20_day_vol_MA': latest_data['20_day_vol_MA']
                        
                    })
                return pd.DataFrame(technical_data)

            macd_signal_list = []
            moving_average_tickers = []
            bollinger_low_cross_tickers = []
            volume_increase_tickers = []

            progress_bar = st.progress(0)
            progress_step = 1 / len(tickers)

            for i, ticker in enumerate(tickers):
                progress_bar.progress((i + 1) * progress_step)
                data = yf.download(ticker, start=start_date, end=end_date)
                if data.empty:
                    continue
                data = calculate_indicators(data)
                if submenu == "MACD":
                    data = calculate_macd(data)
                    occurrence_date = check_macd_signal(data)
                    if occurrence_date:
                        macd_signal_list.append((ticker, occurrence_date))
                elif submenu == "Moving Average":
                    data = calculate_ema(data)
                    occurrence_date = check_moving_average_crossover(data)
                    if occurrence_date:
                        moving_average_tickers.append((ticker, occurrence_date))
                elif submenu == "Bollinger Bands":
                    occurrence_date = check_bollinger_low_cross(data)
                    if occurrence_date:
                        bollinger_low_cross_tickers.append((ticker, occurrence_date))
                elif submenu == "Volume":
                    data = calculate_volume(data)
                    occurrence_date = check_volume_increase(data)
                    if occurrence_date:
                        volume_increase_tickers.append((ticker, occurrence_date))

            df_macd_signal = fetch_latest_data(macd_signal_list)
            df_moving_average_signal = fetch_latest_data(moving_average_tickers)
            df_bollinger_low_cross_signal = fetch_latest_data(bollinger_low_cross_tickers)
            df_volume_increase_signal = fetch_latest_data(volume_increase_tickers)

            st.title("Stock's Based on Selected Strategy")

            if submenu == "MACD":
                st.write("Stocks with MACD > MACD Signal and MACD > 0 in the last 5 days:")
                st.dataframe(df_macd_signal)
            

            elif submenu == "Moving Average":
                st.write("Stocks with 10-day EMA crossing above 20-day EMA in the last 5 days:")
                st.dataframe(df_moving_average_signal)
            

            elif submenu == "Bollinger Bands":
                st.write("Stocks with price crossing below Bollinger Low in the last 5 days:")
                st.dataframe(df_bollinger_low_cross_signal)
            

            elif submenu == "Volume":
                st.write("Stocks with volume above 20-day moving average in the last 5 days:")
                st.dataframe(df_volume_increase_signal)


        elif choice == "Stock Analysis":
            #'Technical Analysis' code---------------------------------------------------------------------------------------------------------------------------------

            st.sidebar.subheader("Stock Analysis")

            # User input for the stock ticker
            ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., BAJAJFINSV.NS): ', 'BAJAJFINSV.NS')


            submenu = st.sidebar.selectbox("Select Analysis Type", ["Financial Analysis", "Technical Analysis", "Sentiment Analysis", "Price Forecast"])



            # Date inputs limited to the last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            if submenu == "Financial Analysis":
                pass

            elif submenu == "Technical Analysis":
                # Define a function to download data
                def download_data(ticker, start_date, end_date):
                    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
                    return df

                # Define a function to calculate William Arbitrage
                def calculate_williams_alligator(df):
                    jaw_length = 13
                    teeth_length = 8
                    lips_length = 5

                    df['Jaw'] = df['Close'].shift(jaw_length).rolling(window=jaw_length).mean()
                    df['Teeth'] = df['Close'].shift(teeth_length).rolling(window=teeth_length).mean()
                    df['Lips'] = df['Close'].shift(lips_length).rolling(window=lips_length).mean()

                    return df

                # Define a function to calculate technical indicators
                def calculate_indicators(df):
                    df['CMO'] = ta.cmo(df['Close'], length=14)
                    
                    keltner = ta.kc(df['High'], df['Low'], df['Close'])
                    df['Keltner_High'] = keltner['KCUe_20_2']
                    df['Keltner_Low'] = keltner['KCLe_20_2']
                    
                    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
                    df['Ultimate_Oscillator'] = ta.uo(df['High'], df['Low'], df['Close'])
                    
                    kvo = ta.kvo(df['High'], df['Low'], df['Close'], df['Volume'])
                    df['Klinger'] = kvo['KVO_34_55_13']
                    
                    donchian = ta.donchian(df['High'], df['Low'])
                    df['Donchian_High'] = donchian['DCU_20_20']
                    df['Donchian_Low'] = donchian['DCL_20_20']
                    
                    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume']).astype(float)
                    
                    distance_moved = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
                    box_ratio = (df['Volume'] / 1e8) / (df['High'] - df['Low'])
                    emv = distance_moved / box_ratio
                    df['Ease_of_Movement'] = emv.rolling(window=14).mean()
                    
                    df['Chaikin_MF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
                    
                    df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'])
                    
                    trix = ta.trix(df['Close'])
                    df['Trix'] = trix['TRIX_30_9']
                    df['Trix_Signal'] = trix['TRIXs_30_9']
                    
                    vortex = ta.vortex(df['High'], df['Low'], df['Close'])
                    df['Vortex_Pos'] = vortex['VTXP_14']
                    df['Vortex_Neg'] = vortex['VTXM_14']
                    
                    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3.0)
                    df['SuperTrend'] = supertrend['SUPERT_7_3.0']
                    
                    df['RVI'] = ta.rvi(df['High'], df['Low'], df['Close'])
                    df['RVI_Signal'] = ta.ema(df['RVI'], length=14)
                    
                    bull_power = df['High'] - ta.ema(df['Close'], length=13)
                    bear_power = df['Low'] - ta.ema(df['Close'], length=13)
                    df['Elder_Ray_Bull'] = bull_power
                    df['Elder_Ray_Bear'] = bear_power
                    
                    wad = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
                    df['Williams_AD'] = wad
                    
                    # Darvas Box Theory
                    df['Darvas_High'] = df['High'].rolling(window=20).max()
                    df['Darvas_Low'] = df['Low'].rolling(window=20).min()
                    
                    # Volume Profile calculation
                    df['Volume_Profile'] = df.groupby(pd.cut(df['Close'], bins=20))['Volume'].transform('sum')

                    # Additional technical indicators
                    df['5_day_EMA'] = df['Close'].ewm(span=5, adjust=False).mean()
                    df['10_day_EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
                    df['20_day_EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
                    df['12_day_EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
                    df['26_day_EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = df['12_day_EMA'] - df['26_day_EMA']
                    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
                    
                    delta = df['Close'].diff(1)
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    low_14 = df['Low'].rolling(window=14).min()
                    high_14 = df['High'].rolling(window=14).max()
                    df['Stochastic_%K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
                    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
                    
                    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
                    
                    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
                    df['A/D_line'] = (clv * df['Volume']).fillna(0).cumsum()
                    
                    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
                    df['5_day_Volume_MA'] = df['Volume'].rolling(window=5).mean()
                    df['10_day_Volume_MA'] = df['Volume'].rolling(window=10).mean()
                    df['20_day_Volume_MA'] = df['Volume'].rolling(window=20).mean()
                    df['20_day_SMA'] = df['Close'].rolling(window=20).mean()
                    df['Std_Dev'] = df['Close'].rolling(window=20).std()
                    df['BB_High'] = df['20_day_SMA'] + (df['Std_Dev'] * 2)
                    df['BB_Low'] = df['20_day_SMA'] - (df['Std_Dev'] * 2)
                    
                    high_low = df['High'] - df['Low']
                    high_close = np.abs(df['High'] - df['Close'].shift())
                    low_close = np.abs(df['Low'] - df['Close'].shift())
                    tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum)
                    df['ATR'] = tr.rolling(window=14).mean()
                    
                    # Parabolic SAR calculation
                    df['Parabolic_SAR'] = calculate_parabolic_sar(df)
                    
                    # ADX calculation
                    df['ADX'] = calculate_adx(df)
                    
                    # Ichimoku Cloud calculation
                    df['Ichimoku_conv'], df['Ichimoku_base'], df['Ichimoku_A'], df['Ichimoku_B'] = calculate_ichimoku(df)
                    
                    # Other indicators
                    df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
                    df['DPO'] = df['Close'] - df['Close'].shift(21).rolling(window=21).mean()
                    df['Williams_%R'] = (high_14 - df['Close']) / (high_14 - low_14) * -100
                    df['McClellan_Oscillator'] = df['Close'].ewm(span=19, adjust=False).mean() - df['Close'].ewm(span=39, adjust=False).mean()
                    
                    advances = (df['Close'] > df['Open']).astype(int)
                    declines = (df['Close'] < df['Open']).astype(int)
                    df['TRIN'] = (advances.rolling(window=14).sum() / declines.rolling(window=14).sum()) / (df['Volume'].rolling(window=14).mean() / df['Volume'].rolling(window=14).mean())
                    df['Price_to_Volume'] = df['Close'] / df['Volume']
                    df['Trend_Line'] = df['Close'].rolling(window=30).mean()
                    
                    # Pivot Points calculation
                    df['Pivot_Point'], df['Resistance_1'], df['Support_1'], df['Resistance_2'], df['Support_2'], df['Resistance_3'], df['Support_3'] = calculate_pivot_points(df)
                    
                    # Fibonacci Levels calculation
                    df = calculate_fibonacci_levels(df)
                    
                    # Gann Levels calculation
                    df = calculate_gann_levels(df)
                    
                    # Advance Decline Line calculation
                    df['Advance_Decline_Line'] = advances.cumsum() - declines.cumsum()
                    
                    # William Arbitrage calculation
                    df = calculate_williams_alligator(df)
                    
                    return df

                def calculate_parabolic_sar(df):
                    af = 0.02
                    uptrend = True
                    df['Parabolic_SAR'] = np.nan
                    ep = df['Low'][0] if uptrend else df['High'][0]
                    df['Parabolic_SAR'].iloc[0] = df['Close'][0]
                    for i in range(1, len(df)):
                        if uptrend:
                            df['Parabolic_SAR'].iloc[i] = df['Parabolic_SAR'].iloc[i - 1] + af * (ep - df['Parabolic_SAR'].iloc[i - 1])
                            if df['Low'].iloc[i] < df['Parabolic_SAR'].iloc[i]:
                                uptrend = False
                                df['Parabolic_SAR'].iloc[i] = ep
                                af = 0.02
                                ep = df['High'].iloc[i]
                        else:
                            df['Parabolic_SAR'].iloc[i] = df['Parabolic_SAR'].iloc[i - 1] + af * (ep - df['Parabolic_SAR'].iloc[i - 1])
                            if df['High'].iloc[i] > df['Parabolic_SAR'].iloc[i]:
                                uptrend = True
                                df['Parabolic_SAR'].iloc[i] = ep
                                af = 0.02
                                ep = df['Low'].iloc[i]
                    return df['Parabolic_SAR']

                def calculate_adx(df):
                    plus_dm = df['High'].diff()
                    minus_dm = df['Low'].diff()
                    plus_dm[plus_dm < 0] = 0
                    minus_dm[minus_dm > 0] = 0
                    tr = pd.concat([df['High'] - df['Low'], 
                                    (df['High'] - df['Close'].shift()).abs(), 
                                    (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean()
                    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
                    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
                    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(alpha=1/14).mean()
                    return adx

                def calculate_ichimoku(df):
                    high_9 = df['High'].rolling(window=9).max()
                    low_9 = df['Low'].rolling(window=9).min()
                    high_26 = df['High'].rolling(window=26).max()
                    low_26 = df['Low'].rolling(window=26).min()
                    high_52 = df['High'].rolling(window=52).max()
                    low_52 = df['Low'].rolling(window=52).min()
                    ichimoku_conv = (high_9 + low_9) / 2
                    ichimoku_base = (high_26 + low_26) / 2
                    ichimoku_a = ((ichimoku_conv + ichimoku_base) / 2).shift(26)
                    ichimoku_b = ((high_52 + low_52) / 2).shift(26)
                    return ichimoku_conv, ichimoku_base, ichimoku_a, ichimoku_b

                def calculate_pivot_points(df):
                    pivot = (df['High'] + df['Low'] + df['Close']) / 3
                    resistance_1 = (2 * pivot) - df['Low']
                    support_1 = (2 * pivot) - df['High']
                    resistance_2 = pivot + (df['High'] - df['Low'])
                    support_2 = pivot - (df['High'] - df['Low'])
                    resistance_3 = df['High'] + 2 * (pivot - df['Low'])
                    support_3 = df['Low'] - 2 * (df['High'] - pivot)
                    return pivot, resistance_1, support_1, resistance_2, support_2, resistance_3, support_3

                def calculate_fibonacci_levels(df):
                    high = df['High'].max()
                    low = df['Low'].min()
                    diff = high - low
                    df['Fib_0.0'] = high
                    df['Fib_0.236'] = high - 0.236 * diff
                    df['Fib_0.382'] = high - 0.382 * diff
                    df['Fib_0.5'] = high - 0.5 * diff
                    df['Fib_0.618'] = high - 0.618 * diff
                    df['Fib_1.0'] = low
                    return df

                def calculate_gann_levels(df):
                    high = df['High'].max()
                    low = df['Low'].min()
                    diff = high - low
                    df['Gann_0.25'] = low + 0.25 * diff
                    df['Gann_0.5'] = low + 0.5 * diff
                    df['Gann_0.75'] = low + 0.75 * diff
                    return df

                # Function to calculate the scores based on the provided criteria
                def calculate_scores(data):
                    scores = {
                        'Trend': 0,
                        'Momentum': 0,
                        'Volatility': 0,
                        'Volume': 0,
                        'Support_Resistance': 0
                    }
                    details = {
                        'Trend': "",
                        'Momentum': "",
                        'Volatility': "",
                        'Volume': "",
                        'Support_Resistance': ""
                    }
                    
                    # Trend Indicators
                    trend_score = 0
                    if data['5_day_EMA'].iloc[-1] > data['10_day_EMA'].iloc[-1] > data['20_day_EMA'].iloc[-1]:
                        trend_score += 2
                        details['Trend'] += "EMA: Strong Bullish; "
                    elif data['5_day_EMA'].iloc[-1] > data['10_day_EMA'].iloc[-1] > data['20_day_EMA'].iloc[-1] and abs(data['5_day_EMA'].iloc[-1] - data['20_day_EMA'].iloc[-1]) < 0.5:
                        trend_score += 1.75
                        details['Trend'] += "EMA: Moderate Bullish; "
                    elif data['5_day_EMA'].iloc[-1] > data['10_day_EMA'].iloc[-1] and data['10_day_EMA'].iloc[-1] < data['20_day_EMA'].iloc[-1]:
                        trend_score += 1.5
                        details['Trend'] += "EMA: Moderate Bullish; "
                    elif data['5_day_EMA'].iloc[-1] < data['10_day_EMA'].iloc[-1] > data['20_day_EMA'].iloc[-1]:
                        trend_score += 1
                        details['Trend'] += "EMA: Neutral; "
                    elif data['5_day_EMA'].iloc[-1] < data['10_day_EMA'].iloc[-1] < data['20_day_EMA'].iloc[-1]:
                        trend_score += 0
                        details['Trend'] += "EMA: Bearish; "
                    
                    # MACD Histogram
                    macd_hist = data['MACD_hist'].iloc[-1]
                    if macd_hist > 0 and macd_hist > data['MACD_hist'].iloc[-2]:
                        trend_score += 2
                        details['Trend'] += "MACD: Strong Bullish; "
                    elif macd_hist > 0:
                        trend_score += 1.75
                        details['Trend'] += "MACD: Moderate Bullish; "
                    elif macd_hist > 0 and macd_hist < data['MACD_hist'].iloc[-2]:
                        trend_score += 1.5
                        details['Trend'] += "MACD: Mild Bullish; "
                    elif macd_hist < 0 and macd_hist > data['MACD_hist'].iloc[-2]:
                        trend_score += 1
                        details['Trend'] += "MACD: Mild Bearish; "
                    elif macd_hist < 0:
                        trend_score += 0.75
                        details['Trend'] += "MACD: Neutral; "
                    elif macd_hist < 0 and macd_hist < data['MACD_hist'].iloc[-2]:
                        trend_score += 0
                        details['Trend'] += "MACD: Strong Bearish; "
                    
                    # Ichimoku
                    ichimoku_conv = data['Ichimoku_conv'].iloc[-1]
                    ichimoku_base = data['Ichimoku_base'].iloc[-1]
                    ichimoku_a = data['Ichimoku_A'].iloc[-1]
                    ichimoku_b = data['Ichimoku_B'].iloc[-1]
                    if ichimoku_conv > ichimoku_base and ichimoku_conv > ichimoku_a and ichimoku_conv > ichimoku_b:
                        trend_score += 1
                        details['Trend'] += "Ichimoku: Strong Bullish; "
                    elif ichimoku_conv > ichimoku_base and (ichimoku_conv > ichimoku_a or ichimoku_conv > ichimoku_b):
                        trend_score += 0.75
                        details['Trend'] += "Ichimoku: Moderate Bullish; "
                    elif ichimoku_conv > ichimoku_base:
                        trend_score += 0.5
                        details['Trend'] += "Ichimoku: Mild Bullish; "
                    elif ichimoku_conv < ichimoku_base and (ichimoku_conv > ichimoku_a or ichimoku_conv > ichimoku_b):
                        trend_score += 0.5
                        details['Trend'] += "Ichimoku: Neutral; "
                    elif ichimoku_conv < ichimoku_base:
                        trend_score += 0.25
                        details['Trend'] += "Ichimoku: Mild Bearish; "
                    else:
                        trend_score += 0
                        details['Trend'] += "Ichimoku: Bearish; "
                    
                    # Parabolic SAR
                    psar = data['Parabolic_SAR'].iloc[-1]
                    price = data['Close'].iloc[-1]
                    if psar < price and psar > data['Parabolic_SAR'].iloc[-2]:
                        trend_score += 1
                        details['Trend'] += "Parabolic SAR: Strong Bullish; "
                    elif psar < price:
                        trend_score += 0.75
                        details['Trend'] += "Parabolic SAR: Moderate Bullish; "
                    elif psar < price:
                        trend_score += 0.5
                        details['Trend'] += "Parabolic SAR: Mild Bullish; "
                    elif psar < price:
                        trend_score += 0.5
                        details['Trend'] += "Parabolic SAR: Neutral; "
                    else:
                        trend_score += 0
                        details['Trend'] += "Parabolic SAR: Strong Bearish; "
                    
                    # SuperTrend
                    supertrend = data['SuperTrend'].iloc[-1]
                    if supertrend < price:
                        trend_score += 1
                        details['Trend'] += "SuperTrend: Strong Bullish; "
                    elif supertrend < price:
                        trend_score += 0.75
                        details['Trend'] += "SuperTrend: Moderate Bullish; "
                    else:
                        trend_score += 0
                        details['Trend'] += "SuperTrend: Bearish; "
                    
                    # Donchian Channels
                    donchian_high = data['Donchian_High'].iloc[-1]
                    donchian_low = data['Donchian_Low'].iloc[-1]
                    if price > donchian_high:
                        trend_score += 1
                        details['Trend'] += "Donchian Channels: Strong Bullish; "
                    elif price > donchian_high:
                        trend_score += 0.75
                        details['Trend'] += "Donchian Channels: Moderate Bullish; "
                    elif price > donchian_low:
                        trend_score += 0.5
                        details['Trend'] += "Donchian Channels: Mild Bullish; "
                    elif price > donchian_low:
                        trend_score += 0.5
                        details['Trend'] += "Donchian Channels: Neutral; "
                    else:
                        trend_score += 0.25
                        details['Trend'] += "Donchian Channels: Mild Bearish; "
                    
                    # Vortex Indicator
                    vortex_pos = data['Vortex_Pos'].iloc[-1]
                    vortex_neg = data['Vortex_Neg'].iloc[-1]
                    if vortex_pos > vortex_neg and vortex_pos > data['Vortex_Pos'].iloc[-2]:
                        trend_score += 1
                        details['Trend'] += "Vortex: Strong Bullish; "
                    elif vortex_pos > vortex_neg:
                        trend_score += 0.75
                        details['Trend'] += "Vortex: Moderate Bullish; "
                    elif vortex_pos > vortex_neg:
                        trend_score += 0.5
                        details['Trend'] += "Vortex: Mild Bullish; "
                    elif vortex_pos < vortex_neg and vortex_pos > data['Vortex_Pos'].iloc[-2]:
                        trend_score += 0.5
                        details['Trend'] += "Vortex: Neutral; "
                    elif vortex_pos < vortex_neg:
                        trend_score += 0.25
                        details['Trend'] += "Vortex: Mild Bearish; "
                    else:
                        trend_score += 0
                        details['Trend'] += "Vortex: Bearish; "
                    
                    # ADX
                    adx = data['ADX'].iloc[-1]
                    if adx > 25 and adx > data['ADX'].iloc[-2]:
                        trend_score += 1
                        details['Trend'] += "ADX: Strong Trend; "
                    elif adx > 25:
                        trend_score += 0.75
                        details['Trend'] += "ADX: Moderate Trend; "
                    elif adx > 20:
                        trend_score += 0.5
                        details['Trend'] += "ADX: Building Trend; "
                    else:
                        trend_score += 0.25
                        details['Trend'] += "ADX: Weak Trend; "
                    
                    scores['Trend'] = trend_score / 11  # Normalize to 1

                    # Momentum Indicators
                    momentum_score = 0
                    rsi = data['RSI'].iloc[-1]
                    if rsi > 70:
                        momentum_score += 0
                        details['Momentum'] += "RSI: Overbought (Bearish); "
                    elif rsi > 60:
                        momentum_score += 0.25
                        details['Momentum'] += "RSI: Mildly Overbought (Bearish); "
                    elif rsi > 50:
                        momentum_score += 0.75
                        details['Momentum'] += "RSI: Mild Bullish; "
                    elif rsi > 40:
                        momentum_score += 0.5
                        details['Momentum'] += "RSI: Neutral; "
                    elif rsi > 30:
                        momentum_score += 0.25
                        details['Momentum'] += "RSI: Mild Bearish; "
                    else:
                        momentum_score += 1
                        details['Momentum'] += "RSI: Oversold (Bullish); "

                    # Stochastic %K and %D
                    stoch_k = data['Stochastic_%K'].iloc[-1]
                    stoch_d = data['Stochastic_%D'].iloc[-1]
                    if stoch_k > stoch_d and stoch_k > 80:
                        momentum_score += 0
                        details['Momentum'] += "Stochastic: Overbought (Bearish); "
                    elif stoch_k > stoch_d and stoch_k > 50:
                        momentum_score += 0.75
                        details['Momentum'] += "Stochastic: Bullish; "
                    elif stoch_k > stoch_d:
                        momentum_score += 0.5
                        details['Momentum'] += "Stochastic: Neutral Bullish; "
                    elif stoch_k < stoch_d and stoch_k < 20:
                        momentum_score += 1
                        details['Momentum'] += "Stochastic: Oversold (Bullish); "
                    elif stoch_k < stoch_d:
                        momentum_score += 0.5
                        details['Momentum'] += "Stochastic: Neutral Bearish; "

                    # Rate of Change (ROC)
                    roc = data['ROC'].iloc[-1]
                    if roc > 10:
                        momentum_score += 1
                        details['Momentum'] += "ROC: Strong Bullish; "
                    elif roc > 0:
                        momentum_score += 0.75
                        details['Momentum'] += "ROC: Mild Bullish; "
                    elif roc > -10:
                        momentum_score += 0.25
                        details['Momentum'] += "ROC: Mild Bearish; "
                    else:
                        momentum_score += 0
                        details['Momentum'] += "ROC: Strong Bearish; "

                    # Detrended Price Oscillator (DPO)
                    dpo = data['DPO'].iloc[-1]
                    if dpo > 1:
                        momentum_score += 1
                        details['Momentum'] += "DPO: Strong Bullish; "
                    elif dpo > 0:
                        momentum_score += 0.75
                        details['Momentum'] += "DPO: Mild Bullish; "
                    elif dpo > -1:
                        momentum_score += 0.25
                        details['Momentum'] += "DPO: Mild Bearish; "
                    else:
                        momentum_score += 0
                        details['Momentum'] += "DPO: Strong Bearish; "

                    # Williams %R
                    williams_r = data['Williams_%R'].iloc[-1]
                    if williams_r > -20:
                        momentum_score += 0
                        details['Momentum'] += "Williams %R: Overbought (Bearish); "
                    elif williams_r > -50:
                        momentum_score += 0.25
                        details['Momentum'] += "Williams %R: Neutral Bearish; "
                    elif williams_r > -80:
                        momentum_score += 0.5
                        details['Momentum'] += "Williams %R: Neutral Bullish; "
                    else:
                        momentum_score += 1
                        details['Momentum'] += "Williams %R: Oversold (Bullish); "

                    # Chande Momentum Oscillator (CMO)
                    cmo = data['CMO'].iloc[-1]
                    if cmo > 50:
                        momentum_score += 0
                        details['Momentum'] += "CMO: Overbought (Bearish); "
                    elif cmo > 0:
                        momentum_score += 0.75
                        details['Momentum'] += "CMO: Bullish; "
                    elif cmo > -50:
                        momentum_score += 0.5
                        details['Momentum'] += "CMO: Neutral; "
                    else:
                        momentum_score += 1
                        details['Momentum'] += "CMO: Oversold (Bullish); "

                    # Commodity Channel Index (CCI)
                    cci = data['CCI'].iloc[-1]
                    if cci > 200:
                        momentum_score += 1
                        details['Momentum'] += "CCI: Strong Bullish; "
                    elif cci > 100:
                        momentum_score += 0.75
                        details['Momentum'] += "CCI: Mild Bullish; "
                    elif cci > -100:
                        momentum_score += 0.5
                        details['Momentum'] += "CCI: Neutral; "
                    elif cci > -200:
                        momentum_score += 0.25
                        details['Momentum'] += "CCI: Mild Bearish; "
                    else:
                        momentum_score += 0
                        details['Momentum'] += "CCI: Strong Bearish; "

                    # Relative Vigor Index (RVI)
                    rvi = data['RVI'].iloc[-1]
                    rvi_signal = data['RVI_Signal'].iloc[-1]
                    if rvi > rvi_signal:
                        momentum_score += 1
                        details['Momentum'] += "RVI: Strong Bullish; "
                    elif rvi > rvi_signal:
                        momentum_score += 0.75
                        details['Momentum'] += "RVI: Mild Bullish; "
                    elif rvi < rvi_signal:
                        momentum_score += 0.25
                        details['Momentum'] += "RVI: Mild Bearish; "
                    else:
                        momentum_score += 0
                        details['Momentum'] += "RVI: Strong Bearish; "

                    # Ultimate Oscillator
                    uo = data['Ultimate_Oscillator'].iloc[-1]
                    if uo > 70:
                        momentum_score += 0
                        details['Momentum'] += "Ultimate Oscillator: Overbought (Bearish); "
                    elif uo > 60:
                        momentum_score += 0.25
                        details['Momentum'] += "Ultimate Oscillator: Mild Bearish; "
                    elif uo > 50:
                        momentum_score += 0.75
                        details['Momentum'] += "Ultimate Oscillator: Neutral Bullish; "
                    elif uo > 40:
                        momentum_score += 0.25
                        details['Momentum'] += "Ultimate Oscillator: Neutral Bearish; "
                    elif uo > 30:
                        momentum_score += 0.75
                        details['Momentum'] += "Ultimate Oscillator: Mild Bullish; "
                    else:
                        momentum_score += 1
                        details['Momentum'] += "Ultimate Oscillator: Oversold (Bullish); "

                    # Trix and Trix Signal
                    trix = data['Trix'].iloc[-1]
                    trix_signal = data['Trix_Signal'].iloc[-1]
                    if trix > trix_signal:
                        momentum_score += 1
                        details['Momentum'] += "Trix: Strong Bullish; "
                    elif trix > trix_signal:
                        momentum_score += 0.75
                        details['Momentum'] += "Trix: Mild Bullish; "
                    elif trix < trix_signal:
                        momentum_score += 0.25
                        details['Momentum'] += "Trix: Mild Bearish; "
                    else:
                        momentum_score += 0
                        details['Momentum'] += "Trix: Strong Bearish; "

                    # Klinger Oscillator
                    klinger = data['Klinger'].iloc[-1]
                    if klinger > 50:
                        momentum_score += 1
                        details['Momentum'] += "Klinger: Strong Bullish; "
                    elif klinger > 0:
                        momentum_score += 0.75
                        details['Momentum'] += "Klinger: Mild Bullish; "
                    elif klinger > -50:
                        momentum_score += 0.25
                        details['Momentum'] += "Klinger: Mild Bearish; "
                    else:
                        momentum_score += 0
                        details['Momentum'] += "Klinger: Strong Bearish; "

                    scores['Momentum'] = momentum_score / 13  # Normalize to 1

                    # Volatility Indicators
                    volatility_score = 0
                    atr = data['ATR'].iloc[-1]
                    std_dev = data['Std_Dev'].iloc[-1]
                    bb_high = data['BB_High'].iloc[-1]
                    bb_low = data['BB_Low'].iloc[-1]
                    sma_20 = data['20_day_SMA'].iloc[-1]
                    keltner_high = data['Keltner_High'].iloc[-1]
                    keltner_low = data['Keltner_Low'].iloc[-1]

                    # ATR
                    if atr > 1.5 * data['ATR'].rolling(window=50).mean().iloc[-1]:
                        volatility_score += 0.5
                        details['Volatility'] += "ATR: High volatility; "
                    elif atr > 1.2 * data['ATR'].rolling(window=50).mean().iloc[-1]:
                        volatility_score += 0.4
                        details['Volatility'] += "ATR: Moderate volatility; "
                    else:
                        volatility_score += 0.3
                        details['Volatility'] += "ATR: Low volatility; "

                    # Standard Deviation
                    if std_dev > 1.5 * data['Std_Dev'].rolling(window=50).mean().iloc[-1]:
                        volatility_score += 0.5
                        details['Volatility'] += "Std Dev: High volatility; "
                    elif std_dev > 1.2 * data['Std_Dev'].rolling(window=50).mean().iloc[-1]:
                        volatility_score += 0.4
                        details['Volatility'] += "Std Dev: Moderate volatility; "
                    else:
                        volatility_score += 0.3
                        details['Volatility'] += "Std Dev: Low volatility; "

                    # Bollinger Bands
                    if price > bb_high:
                        volatility_score += 0
                        details['Volatility'] += "BB: Strong Overbought (Bearish); "
                    elif price > bb_high:
                        volatility_score += 0.25
                        details['Volatility'] += "BB: Mild Overbought (Bearish); "
                    elif price < bb_low:
                        volatility_score += 1
                        details['Volatility'] += "BB: Strong Oversold (Bullish); "
                    elif price < bb_low:
                        volatility_score += 0.75
                        details['Volatility'] += "BB: Mild Oversold (Bullish); "
                    else:
                        volatility_score += 0.5
                        details['Volatility'] += "BB: Neutral; "

                    # 20-day SMA
                    if price > sma_20:
                        volatility_score += 1
                        details['Volatility'] += "20-day SMA: Strong Bullish; "
                    elif price > sma_20:
                        volatility_score += 0.75
                        details['Volatility'] += "20-day SMA: Mild Bullish; "
                    elif price < sma_20:
                        volatility_score += 0.25
                        details['Volatility'] += "20-day SMA: Mild Bearish; "
                    else:
                        volatility_score += 0
                        details['Volatility'] += "20-day SMA: Strong Bearish; "

                    # Keltner Channels
                    if price > keltner_high:
                        volatility_score += 0
                        details['Volatility'] += "Keltner: Strong Overbought (Bearish); "
                    elif price > keltner_high:
                        volatility_score += 0.25
                        details['Volatility'] += "Keltner: Mild Overbought (Bearish); "
                    elif price < keltner_low:
                        volatility_score += 1
                        details['Volatility'] += "Keltner: Strong Oversold (Bullish); "
                    elif price < keltner_low:
                        volatility_score += 0.75
                        details['Volatility'] += "Keltner: Mild Oversold (Bullish); "
                    else:
                        volatility_score += 0.5
                        details['Volatility'] += "Keltner: Neutral; "
                    
                    scores['Volatility'] = volatility_score / 5  # Normalize to 1

                    # Volume Indicators
                    volume_score = 0
                    obv = data['OBV'].iloc[-1]
                    ad_line = data['A/D_line'].iloc[-1]
                    price_to_volume = data['Price_to_Volume'].iloc[-1]
                    trin = data['TRIN'].iloc[-1]
                    advance_decline_line = data['Advance_Decline_Line'].iloc[-1]
                    mcclellan_oscillator = data['McClellan_Oscillator'].iloc[-1]
                    volume_profile = data['Volume_Profile'].iloc[-1]
                    cmf = data['Chaikin_MF'].iloc[-1]
                    williams_ad = data['Williams_AD'].iloc[-1]
                    ease_of_movement = data['Ease_of_Movement'].iloc[-1]
                    mfi = data['MFI'].iloc[-1]
                    elder_ray_bull = data['Elder_Ray_Bull'].iloc[-1]
                    elder_ray_bear = data['Elder_Ray_Bear'].iloc[-1]
                    vwap = data['VWAP'].iloc[-1]

                    if obv > data['OBV'].iloc[-2]:
                        volume_score += 1
                        details['Volume'] += "OBV: Increasing sharply; "
                    elif obv < data['OBV'].iloc[-2]:
                        volume_score += 0
                        details['Volume'] += "OBV: Decreasing; "

                    if ad_line > data['A/D_line'].iloc[-2]:
                        volume_score += 1
                        details['Volume'] += "A/D Line: Increasing sharply; "
                    elif ad_line < data['A/D_line'].iloc[-2]:
                        volume_score += 0
                        details['Volume'] += "A/D Line: Decreasing; "

                    # Add more volume indicators scoring
                    # Price to Volume
                    if price_to_volume > data['Price_to_Volume'].mean():
                        volume_score += 0.75
                        details['Volume'] += "Price to Volume: Increasing; "
                    elif price_to_volume < data['Price_to_Volume'].mean():
                        volume_score += 0.25
                        details['Volume'] += "Price to Volume: Decreasing; "
                    else:
                        volume_score += 0.5
                        details['Volume'] += "Price to Volume: Neutral; "

                    # TRIN (Arms Index)
                    if trin < 0.8:
                        volume_score += 1
                        details['Volume'] += "TRIN: Strong Bullish; "
                    elif trin < 1:
                        volume_score += 0.75
                        details['Volume'] += "TRIN: Mild Bullish; "
                    elif trin < 1.2:
                        volume_score += 0.5
                        details['Volume'] += "TRIN: Neutral; "
                    elif trin < 1.5:
                        volume_score += 0.25
                        details['Volume'] += "TRIN: Mild Bearish; "
                    else:
                        volume_score += 0
                        details['Volume'] += "TRIN: Strong Bearish; "

                    # Advance/Decline Line
                    if advance_decline_line > data['Advance_Decline_Line'].iloc[-2]:
                        volume_score += 1
                        details['Volume'] += "Advance/Decline Line: Increasing sharply; "
                    elif advance_decline_line > data['Advance_Decline_Line'].iloc[-2]:
                        volume_score += 0.75
                        details['Volume'] += "Advance/Decline Line: Increasing slowly; "
                    elif advance_decline_line < data['Advance_Decline_Line'].iloc[-2]:
                        volume_score += 0.25
                        details['Volume'] += "Advance/Decline Line: Decreasing slowly; "
                    else:
                        volume_score += 0
                        details['Volume'] += "Advance/Decline Line: Decreasing sharply; "

                    # McClellan Oscillator
                    if mcclellan_oscillator > 50:
                        volume_score += 1
                        details['Volume'] += "McClellan Oscillator: Strong Bullish; "
                    elif mcclellan_oscillator > 0:
                        volume_score += 0.75
                        details['Volume'] += "McClellan Oscillator: Mild Bullish; "
                    elif mcclellan_oscillator < -50:
                        volume_score += 0.25
                        details['Volume'] += "McClellan Oscillator: Mild Bearish; "
                    else:
                        volume_score += 0
                        details['Volume'] += "McClellan Oscillator: Strong Bearish; "

                    # Volume Profile
                    if volume_profile > data['Volume_Profile'].mean():
                        volume_score += 0.75
                        details['Volume'] += "Volume Profile: High volume nodes well above support; "
                    else:
                        volume_score += 0
                        details['Volume'] += "Volume Profile: High volume nodes at resistance; "

                    # Chaikin Money Flow (CMF)
                    if cmf > 0.2:
                        volume_score += 1
                        details['Volume'] += "CMF: Strong Bullish; "
                    elif cmf > 0:
                        volume_score += 0.75
                        details['Volume'] += "CMF: Mild Bullish; "
                    elif cmf < -0.2:
                        volume_score += 0.25
                        details['Volume'] += "CMF: Mild Bearish; "
                    else:
                        volume_score += 0
                        details['Volume'] += "CMF: Strong Bearish; "

                    # Williams Accumulation/Distribution
                    if williams_ad > data['Williams_AD'].iloc[-2]:
                        volume_score += 1
                        details['Volume'] += "Williams AD: Increasing sharply; "
                    elif williams_ad < data['Williams_AD'].iloc[-2]:
                        volume_score += 0.25
                        details['Volume'] += "Williams AD: Decreasing slowly; "
                    else:
                        volume_score += 0
                        details['Volume'] += "Williams AD: Decreasing sharply; "

                    # Ease of Movement
                    if ease_of_movement > data['Ease_of_Movement'].mean():
                        volume_score += 1
                        details['Volume'] += "Ease of Movement: Positive and increasing; "
                    elif ease_of_movement > 0:
                        volume_score += 0.75
                        details['Volume'] += "Ease of Movement: Positive but flat; "
                    else:
                        volume_score += 0.25
                        details['Volume'] += "Ease of Movement: Negative but flat; "

                    # MFI (Money Flow Index)
                    if mfi > 80:
                        volume_score += 0
                        details['Volume'] += "MFI: Overbought (Bearish); "
                    elif mfi > 70:
                        volume_score += 0.25
                        details['Volume'] += "MFI: Mild Overbought (Bearish); "
                    elif mfi > 50:
                        volume_score += 0.75
                        details['Volume'] += "MFI: Neutral Bullish; "
                    elif mfi > 30:
                        volume_score += 0.5
                        details['Volume'] += "MFI: Neutral; "
                    else:
                        volume_score += 1
                        details['Volume'] += "MFI: Oversold (Bullish); "

                    # Elder-Ray Bull Power and Bear Power
                    if elder_ray_bull > 0 and elder_ray_bull > data['Elder_Ray_Bull'].mean():
                        volume_score += 1
                        details['Volume'] += "Elder Ray Bull Power: Strong Bullish; "
                    elif elder_ray_bull > 0:
                        volume_score += 0.75
                        details['Volume'] += "Elder Ray Bull Power: Mild Bullish; "
                    elif elder_ray_bear < 0 and elder_ray_bear < data['Elder_Ray_Bear'].mean():
                        volume_score += 0
                        details['Volume'] += "Elder Ray Bear Power: Strong Bearish; "
                    else:
                        volume_score += 0.25
                        details['Volume'] += "Elder Ray Bear Power: Mild Bearish; "

                    # VWAP (Volume Weighted Average Price)
                    if price > vwap:
                        volume_score += 1
                        details['Volume'] += "VWAP: Strong Bullish; "
                    elif price > vwap:
                        volume_score += 0.75
                        details['Volume'] += "VWAP: Mild Bullish; "
                    elif price < vwap:
                        volume_score += 0.25
                        details['Volume'] += "VWAP: Mild Bearish; "
                    else:
                        volume_score += 0
                        details['Volume'] += "VWAP: Strong Bearish; "
                    
                    scores['Volume'] = volume_score / 13  # Normalize to 1

                    # Support and Resistance Indicators
                    support_resistance_score = 0
                    price = data['Close'].iloc[-1]
                    pivot_point = data['Pivot_Point'].iloc[-1]
                    resistance_1 = data['Resistance_1'].iloc[-1]
                    support_1 = data['Support_1'].iloc[-1]
                    fib_0 = data['Fib_0.0'].iloc[-1]
                    fib_0_236 = data['Fib_0.236'].iloc[-1]
                    fib_0_382 = data['Fib_0.382'].iloc[-1]
                    fib_0_5 = data['Fib_0.5'].iloc[-1]
                    fib_0_618 = data['Fib_0.618'].iloc[-1]
                    fib_1 = data['Fib_1.0'].iloc[-1]
                    darvas_high = data['Darvas_High'].iloc[-1]
                    darvas_low = data['Darvas_Low'].iloc[-1]

                    if price > pivot_point:
                        support_resistance_score += 1
                        details['Support_Resistance'] += "Pivot Point: Above Pivot Point; "
                    elif price < pivot_point:
                        support_resistance_score += 0
                        details['Support_Resistance'] += "Pivot Point: Below Pivot Point; "

                    if price > resistance_1:
                        support_resistance_score += 0
                        details['Support_Resistance'] += "Support/Resistance: Near Resistance; "
                    elif price < support_1:
                        support_resistance_score += 1
                        details['Support_Resistance'] += "Support/Resistance: Near Support; "

                    # Add Fibonacci Levels scoring
                    if price > fib_0_618:
                        support_resistance_score += 1
                        details['Support_Resistance'] += "Fibonacci: Strong support/resistance; "
                    elif price > fib_0_5:
                        support_resistance_score += 0.75
                        details['Support_Resistance'] += "Fibonacci: Moderate support/resistance; "
                    elif price > fib_0_382:
                        support_resistance_score += 0.5
                        details['Support_Resistance'] += "Fibonacci: Mild support/resistance; "
                    elif price > fib_0_236:
                        support_resistance_score += 0.25
                        details['Support_Resistance'] += "Fibonacci: Weak support/resistance; "
                    else:
                        support_resistance_score += 0.5
                        details['Support_Resistance'] += "Fibonacci: Potential reversal; "

                    # Darvas Box Theory
                    if price > darvas_high:
                        support_resistance_score += 1
                        details['Support_Resistance'] += "Darvas: Strong Bullish; "
                    elif price > darvas_high:
                        support_resistance_score += 0.75
                        details['Support_Resistance'] += "Darvas: Mild Bullish; "
                    elif price < darvas_low:
                        support_resistance_score += 0.25
                        details['Support_Resistance'] += "Darvas: Mild Bearish; "
                    else:
                        support_resistance_score += 0.5
                        details['Support_Resistance'] += "Darvas: Neutral; "

                    scores['Support_Resistance'] = support_resistance_score / 4  # Normalize to 1

                    return scores, details

                # Function to create gauge charts
                def create_gauge(value, title):
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': title},
                        gauge={'axis': {'range': [0, 1]}},
                    ))
                    return fig

                def get_recommendation(overall_score):
                    if overall_score >= 0.8:
                        return "Strong Buy"
                    elif 0.6 <= overall_score < 0.8:
                        return "Buy"
                    elif 0.4 <= overall_score < 0.6:
                        return "Hold"
                    elif 0.2 <= overall_score < 0.4:
                        return "Sell"
                    else:
                        return "Strong Sell"
                    
                # Streamlit App
                st.title('Stock Technical Analysis')
                
                

                # Date inputs
                start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
                end_date = st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=1))

                if ticker:
                    # Download data from Yahoo Finance
                    data = download_data(ticker, start_date, end_date)
                    index_ticker = '^NSEI'  # Nifty 50 index ticker
                    index_data = yf.download(index_ticker, period='1y', interval='1d')

                    # Ensure the index is in datetime format
                    data.index = pd.to_datetime(data.index)
                    index_data.index = pd.to_datetime(index_data.index)

                    # Calculate technical indicators
                    data = calculate_indicators(data)

                    # Option to select table or visualization
                    view_option = st.sidebar.radio("Select View", ('Visualization', 'Analysis'))

                    if view_option == 'Visualization':
                        # Define indicator groups
                        indicator_groups = {
                            "Trend Indicators": ["5_day_EMA", "10_day_EMA", "20_day_EMA", "MACD", "MACD_signal", "MACD_hist", "Trend_Line", "Ichimoku_conv", "Ichimoku_base", "Ichimoku_A", "Ichimoku_B", "Parabolic_SAR", "SuperTrend", "Donchian_High", "Donchian_Low", "Vortex_Pos", "Vortex_Neg", "ADX", "Jaw", "Teeth", "Lips"],
                            "Momentum Indicators": ["RSI", "Stochastic_%K", "Stochastic_%D", "ROC", "DPO", "Williams_%R", "CMO", "CCI", "RVI", "RVI_Signal", "Ultimate_Oscillator", "Trix", "Trix_Signal", "Klinger"],
                            "Volatility": ["ATR", "Std_Dev", "BB_High", "BB_Low","20_day_SMA", "Keltner_High", "Keltner_Low"],
                            "Volume Indicators": ["OBV", "A/D_line", "Price_to_Volume", "TRIN", "Advance_Decline_Line", "McClellan_Oscillator", "Volume_Profile", "Chaikin_MF", "Williams_AD", "Ease_of_Movement", "MFI", "Elder_Ray_Bull", "Elder_Ray_Bear", "VWAP"],
                            "Support and Resistance Levels": ["Pivot_Point", "Resistance_1", "Support_1", "Resistance_2", "Support_2", "Resistance_3", "Support_3", "Fib_0.0", "Fib_0.236", "Fib_0.382", "Fib_0.5", "Fib_0.618", "Fib_1.0", "Darvas_High", "Darvas_Low"],
                            "Other Indicators": ["Relative_Strength", "Performance_vs_Index"]
                        }

                        # Create multiselect options for each indicator group
                        selected_indicators = []
                        for group_name, indicators in indicator_groups.items():
                            with st.expander(group_name):
                                selected_indicators.extend(st.sidebar.multiselect(f'Select {group_name}', indicators))
                        
                        show_candlestick = st.sidebar.checkbox('Heikin-Ashi Candles')

                        def get_macd_hist_colors(macd_hist):
                            colors = []
                            for i in range(1, len(macd_hist)):
                                if macd_hist.iloc[i] > 0:
                                    color = 'green' if macd_hist.iloc[i] > macd_hist.iloc[i - 1] else 'lightgreen'
                                else:
                                    color = 'red' if macd_hist.iloc[i] < macd_hist.iloc[i - 1] else 'lightcoral'
                                colors.append(color)
                            return colors

                        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{}]], 
                                            row_heights=[0.5, 0.3, 0.2], vertical_spacing=0.02)

                        if show_candlestick:
                            fig.add_trace(go.Candlestick(x=data.index,
                                                        open=data['Open'],
                                                        high=data['High'],
                                                        low=data['Low'],
                                                        close=data['Close'],
                                                        name='Candlestick'), row=1, col=1)
                        else:
                            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'), row=1, col=1)

                        for i, indicator in enumerate(selected_indicators):
                            y_axis_name = f'y{i+2}'

                            if indicator == 'MACD_hist':
                                macd_hist_colors = get_macd_hist_colors(data[indicator])
                                fig.add_trace(go.Bar(x=data.index[1:], y=data[indicator][1:], name='MACD Histogram', marker_color=macd_hist_colors), row=1, col=1, secondary_y=True)
                            elif 'Fib' in indicator or 'Gann' in indicator:
                                fig.add_trace(go.Scatter(x=data.index, y=data[indicator], mode='lines', name=indicator, line=dict(dash='dash')), row=1, col=1)
                            else:
                                fig.add_trace(go.Scatter(x=data.index, y=data[indicator], mode='lines', name=indicator), row=1, col=1, secondary_y=True)

                            fig.update_layout(**{
                                f'yaxis{i+2}': go.layout.YAxis(
                                    title=indicator,
                                    overlaying='y',
                                    side='right',
                                    position=1 - (i * 0.05)
                                )
                            })

                        fig.update_layout(
                            title={
                                'text': f'{ticker} Price and Technical Indicators',
                                'y': 0.97,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'
                            },
                            height=900,
                            margin=dict(t=100, b=10, l=50, r=50),
                            yaxis=dict(title='Price'),
                            yaxis2=dict(title='Indicators', overlaying='y', side='right'),
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                rangeselector=dict(
                                    buttons=list([
                                        dict(count=7, label='7d', step='day', stepmode='backward'),
                                        dict(count=14, label='14d', step='day', stepmode='backward'),
                                        dict(count=1, label='1m', step='month', stepmode='backward'),
                                        dict(count=3, label='3m', step='month', stepmode='backward'),
                                        dict(count=6, label='6m', step='month', stepmode='backward'),
                                        dict(count=1, label='1y', step='year', stepmode='backward'),
                                        dict(step='all')
                                    ])
                                ),
                                type='date'
                            ),
                            legend=dict(x=0.5, y=-0.02, orientation='h', xanchor='center', yanchor='top')
                        )

                        fig.update_layout(
                            hovermode='x unified',
                            hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")
                        )

                        st.plotly_chart(fig)
                    else:
                        # Calculate scores
                        scores, details = calculate_scores(data)
                        
                        # Display data table
                        st.write(data)

                        # Create and display gauges and details in two columns
                        for key, value in scores.items():
                            col1, col2 = st.columns([1, 2])
                            with col2:
                                st.plotly_chart(create_gauge(value, key))
                            with col1:
                                st.markdown(f"### {key} Indicators")
                                details_list = details[key].split("; ")
                                for detail in details_list:
                                    if detail:  # Check if detail is not an empty string
                                        st.markdown(f"- {detail}")

                        # Calculate overall weightage score
                        overall_score = np.mean(list(scores.values()))
                        overall_description = f"{overall_score*100:.1f}%"
                        recommendation = get_recommendation(overall_score)
                    
                        col1, col2 = st.columns([1, 2])
                        with col2:
                            st.plotly_chart(create_gauge(overall_score, 'Overall Score'))
                        with col1:
                            st.markdown(f"### Overall Score: {overall_description}")
                            st.markdown(f"<p style='font-size:20px;'>Recommendation: {recommendation}</p>", unsafe_allow_html=True)

            elif submenu == "Sentiment Analysis":

                # Initialize NewsApiClient with your API key
                    newsapi = NewsApiClient(api_key='252b2075083945dfbed8945ddc240a2b')
                    analyzer = SentimentIntensityAnalyzer()

                    def fetch_news(company_name, start_date, end_date):
                        # Fetch news articles related to the company name
                        all_articles = newsapi.get_everything(q=company_name,
                                                            language='en',
                                                            from_param=start_date.strftime('%Y-%m-%d'),
                                                            to=end_date.strftime('%Y-%m-%d'),
                                                            sort_by='publishedAt',
                                                            page_size=50,
                                                            sources='the-times-of-india, financial-express, the-hindu, bloomberg, cnbc')
                        articles = []
                        for article in all_articles['articles']:
                            articles.append({
                                'title': article['title'],
                                'description': article['description'],
                                'url': article['url'],
                                'publishedAt': article['publishedAt'],
                                'source': article['source']['name']
                            })
                        return articles

                    def perform_sentiment_analysis(articles):
                        sentiments = []
                        for article in articles:
                            if article['description']:
                                score = analyzer.polarity_scores(article['description'])
                                article['sentiment'] = score
                                sentiments.append(article)
                            else:
                                article['sentiment'] = {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
                                sentiments.append(article)
                        return sentiments

                    def make_recommendation(sentiments):
                        avg_sentiment = sum([s['sentiment']['compound'] for s in sentiments]) / len(sentiments)
                        if avg_sentiment > 0.1:
                            return "Based on the sentiment analysis, it is recommended to BUY the stock."
                        elif avg_sentiment < -0.1:
                            return "Based on the sentiment analysis, it is recommended to NOT BUY the stock."
                        else:
                            return "Based on the sentiment analysis, it is recommended to HOLD OFF on any action."

                    def count_sentiments(sentiments):
                        positive = sum(1 for s in sentiments if s['sentiment']['compound'] > 0.1)
                        negative = sum(1 for s in sentiments if s['sentiment']['compound'] < -0.1)
                        neutral = len(sentiments) - positive - negative
                        return positive, negative, neutral

                    def generate_wordcloud(text, title):
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.title(title)
                        plt.axis('off')
                        st.pyplot(plt)

                    st.title("Stock News Sentiment Analysis")

                    if ticker:
                        # Fetch company name using yfinance
                        try:
                            company_info = yf.Ticker(ticker)
                            company_name = company_info.info['shortName']
                        except KeyError:
                            company_name = ticker

                        st.write(f"Fetching news for: {company_name}")

                        if company_name:
                            with st.spinner("Fetching news..."):
                                articles = fetch_news(company_name, start_date, end_date)

                            if articles:
                                with st.spinner("Performing sentiment analysis..."):
                                    sentiments = perform_sentiment_analysis(articles)

                                df = pd.DataFrame(sentiments)

                                # Radio button options for Articles and Summary
                                view_option = st.sidebar.radio("View Options", ["Articles", "Summary"])

                                if view_option == "Articles":
                                    st.write("Recent News Articles:")
                                    for i, row in df.iterrows():
                                        st.write(f"**Article {i+1}:** {row['title']}")
                                        st.write(f"**Published At:** {row['publishedAt']}")
                                        st.write(f"**Description:** {row['description']}")
                                        st.write(f"**URL:** {row['url']}")
                                        st.write(f"**Source:** {row['source']}")
                                        st.write(f"**Sentiment Score:** {row['sentiment']['compound']:.2f}")
                                        st.write("---")

                                elif view_option == "Summary":
                                    st.subheader("Sentiment Analysis Summary:")
                                    avg_sentiment = sum(df['sentiment'].apply(lambda x: x['compound'])) / len(df)
                                    st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")

                                    positive, negative, neutral = count_sentiments(sentiments)
                                    st.write(f"Positive Articles: {positive}")
                                    st.write(f"Negative Articles: {negative}")
                                    st.write(f"Neutral Articles: {neutral}")

                                    # Sentiment score bar chart
                                    fig_bar = px.bar(df, x=df.index, y=df['sentiment'].apply(lambda x: x['compound']),
                                                    labels={'x': 'Article', 'y': 'Sentiment Score'},
                                                    title='Sentiment Score per Article')
                                    st.plotly_chart(fig_bar)

                                    # Sentiment trend over time
                                    df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.tz_localize(None)
                                    df.set_index('publishedAt', inplace=True)
                                    df = df.sort_index()

                                    # Aggregate sentiment by day
                                    daily_sentiment = df['sentiment'].apply(lambda x: x['compound']).resample('D').mean()

                                    fig_line = px.line(daily_sentiment.rolling(window=5).mean(),
                                                    labels={'value': 'Sentiment Score', 'index': 'Date'},
                                                    title='Sentiment Trend Over Time')
                                    st.plotly_chart(fig_line)

                                    # Correlate sentiment with stock price
                                    
                                    stock_data = yf.download(ticker, start=start_date, end=end_date)
                                    if not stock_data.empty:
                                        stock_data.index = stock_data.index.tz_localize(None)
                                        stock_data = stock_data[['Close']]
                                        stock_data['Sentiment'] = daily_sentiment
                                        combined_data = pd.concat([stock_data, daily_sentiment], axis=1).dropna()

                                        fig_combined = go.Figure()
                                        fig_combined.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Close'],
                                                                        mode='lines', name='Stock Price'))
                                        fig_combined.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Sentiment'],
                                                                        mode='lines', name='Sentiment Score', yaxis='y2'))

                                        fig_combined.update_layout(
                                            title='Stock Price and Sentiment Over Time',
                                            xaxis_title='Date',
                                            yaxis_title='Stock Price',
                                            yaxis2=dict(title='Sentiment Score', overlaying='y', side='right'),
                                            legend=dict(x=0, y=1, traceorder='normal')
                                        )
                                        st.plotly_chart(fig_combined)

                                    # Sentiment distribution pie chart
                                    
                                    sentiment_counts = pd.Series([positive, negative, neutral], index=['Positive', 'Negative', 'Neutral'])
                                    fig_pie = px.pie(sentiment_counts, values=sentiment_counts, names=sentiment_counts.index, title='Sentiment Distribution')
                                    st.plotly_chart(fig_pie)

                                    # Generate and display word clouds for positive and negative articles
                                    positive_text = ' '.join(df[df['sentiment'].apply(lambda x: x['compound']) > 0.1]['description'].dropna())
                                    negative_text = ' '.join(df[df['sentiment'].apply(lambda x: x['compound']) < -0.1]['description'].dropna())
                                    if positive_text:
                                        st.write("Positive Articles Word Cloud:")
                                        generate_wordcloud(positive_text, 'Positive Articles')
                                    if negative_text:
                                        st.write("Negative Articles Word Cloud:")
                                        generate_wordcloud(negative_text, 'Negative Articles')

                                    recommendation = make_recommendation(sentiments)
                                    st.subheader("Investment Recommendation:")
                                    st.write(recommendation)

                            else:
                                st.write("No news articles found for this company.")
                        else:
                            st.write("Invalid ticker symbol.")

            elif submenu == "Price Forecast":

                # Function to fetch stock data from Yahoo Finance
                def get_stock_data(ticker, start_date, end_date):
                    stock_data = yf.download(ticker, start=start_date, end=end_date)
                    stock_data.reset_index(inplace=True)
                    return stock_data

                # Function to calculate technical indicators
                def calculate_technical_indicators(df):
                    # Moving Average Convergence Divergence (MACD)
                    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
                    df['MACD'] = df['EMA_12'] - df['EMA_26']
                    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
                    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
                                
                    # Relative Strength Index (RSI)
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                                
                    # Bollinger Bands
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                    df['Std_Dev'] = df['Close'].rolling(window=20).std()
                    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
                    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)
                    df['BBW'] = (df['Upper_Band'] - df['Lower_Band']) / df['SMA_20']  # Bollinger Band Width
                                
                    # Exponential Moving Average (EMA)
                    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
                                
                    # Average True Range (ATR)
                    df['High-Low'] = df['High'] - df['Low']
                    df['High-PrevClose'] = np.abs(df['High'] - df['Close'].shift(1))
                    df['Low-PrevClose'] = np.abs(df['Low'] - df['Close'].shift(1))
                    df['True_Range'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
                    df['ATR'] = df['True_Range'].rolling(window=14).mean()
                                
                    # Rate of Change (ROC)
                    df['ROC'] = df['Close'].pct_change(periods=12) * 100
                                
                    # On-Balance Volume (OBV)
                    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
                                
                    return df

                # Functions for Fourier, Wavelet, and Hilbert transforms
                def calculate_fourier(df, n=5):
                    close_fft = np.fft.fft(df['Close'].values)
                    fft_df = pd.DataFrame({'fft': close_fft})
                    fft_df['absolute'] = np.abs(fft_df['fft'])
                    fft_df['angle'] = np.angle(fft_df['fft'])
                    fft_df = fft_df.sort_values(by='absolute', ascending=False).head(n)
                    fft_df = fft_df.reindex(range(len(df)), fill_value=0)  # Ensure it matches the length of the stock data
                    return fft_df['absolute']

                def calculate_wavelet(df):
                    widths = np.arange(1, 31)
                    cwt_matrix = cwt(df['Close'], ricker, widths)
                    max_wavelet = np.max(cwt_matrix, axis=0)
                    return pd.Series(max_wavelet).reindex(range(len(df)), fill_value=0)  # Ensure it matches the length of the stock data

                def calculate_hilbert(df):
                    analytic_signal = hilbert(df['Close'])
                    amplitude_envelope = np.abs(analytic_signal)
                    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                    return pd.Series(amplitude_envelope), pd.Series(instantaneous_phase)

                # Streamlit UI
                st.title("Stock Price Prediction with SARIMA Model")

                start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=1500))
                end_date = st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=1))
            
                forecast_period = st.sidebar.number_input("Forecast Period (days)", value=10, min_value=1, max_value=30)

                # Fetch the data
                stock_data = get_stock_data(ticker, start_date, end_date)

                # Calculate technical indicators
                stock_data = calculate_technical_indicators(stock_data)

                # Calculate Fourier, Wavelet, and Hilbert transforms
                stock_data['Fourier'] = calculate_fourier(stock_data)
                stock_data['Wavelet'] = calculate_wavelet(stock_data)
                stock_data['Hilbert_Amplitude'], stock_data['Hilbert_Phase'] = calculate_hilbert(stock_data)

                # Drop rows with NaN values
                stock_data.dropna(inplace=True)

                # Extract the closing prices and technical indicators
                close_prices = stock_data['Close']
                technical_indicators = stock_data[['MACD', 'Signal_Line', 'MACD_Hist', 'RSI', 'SMA_20', 'Upper_Band', 'Lower_Band', 'BBW', 'EMA_50', 'ATR', 'ROC', 'OBV', 'Fourier', 'Wavelet', 'Hilbert_Amplitude', 'Hilbert_Phase']]

                # Check correlation with close prices
                correlations = technical_indicators.corrwith(close_prices).sort_values()

                # Display correlation as a bar chart
                fig_corr = go.Figure(go.Bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h'
                ))
                fig_corr.update_layout(
                    title="Correlation with Close Prices",
                    xaxis_title="Correlation",
                    yaxis_title="Indicators",
                    yaxis=dict(tickmode='linear')
                )
                st.plotly_chart(fig_corr)

                # Train SARIMA model with exogenous variables (technical indicators)
                sarima_model = auto_arima(
                    close_prices,
                    exogenous=technical_indicators,
                    start_p=1,
                    start_q=1,
                    max_p=3,
                    max_q=3,
                    m=7,  # Weekly seasonality
                    start_P=0,
                    seasonal=True,
                    d=1,
                    D=1,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )

                # Forecasting the next n business days (excluding weekends)
                future_technical_indicators = technical_indicators.tail(forecast_period).values
                forecast = sarima_model.predict(n_periods=forecast_period, exogenous=future_technical_indicators)

                # Generate the forecasted dates excluding weekends
                forecasted_dates = pd.bdate_range(start=stock_data['Date'].iloc[-1], periods=forecast_period + 1)[1:]

                forecasted_df = pd.DataFrame({'Date': forecasted_dates, 'Forecasted_Close': forecast})

                # Plotly interactive figure with time slider
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['Close'],
                    mode='lines',
                    name='Close Price'
                ))

                # Add forecasted prices
                fig.add_trace(go.Scatter(
                    x=forecasted_df['Date'],
                    y=forecasted_df['Forecasted_Close'],
                    mode='lines',
                    name='Forecasted Close Price',
                    line=dict(color='orange')
                ))

                fig.update_layout(
                    title=f'Stock Price Forecast for {ticker}',
                    xaxis_title='Date',
                    yaxis_title='Close Price',
                    legend=dict(x=0.01, y=0.99),
                )

                st.plotly_chart(fig)

                # Display forecasted prices
                st.write("Forecasted Prices for the next {} days:".format(forecast_period))
                st.dataframe(forecasted_df)

                # Display model summary
                st.write("Model Summary:")
                st.text(sarima_model.summary())

        elif choice == "Database Admin Page":
              display_tables()