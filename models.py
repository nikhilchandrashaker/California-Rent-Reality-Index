import pandas as pd, numpy as np, warnings, json
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
RS=42
U=pd.read_pickle('units2.pkl'); U=U[U.model_ready].copy().reset_index(drop=True)
U['y']=np.log(U.rent)
for c in ['has_pool','has_ac','has_fireplace','has_spa']: U[c]=U[c].map({True:1,False:0}).astype(float)
U['log_sqft']=np.log(U.sqft)
und=U.street.str.contains('undisclosed',na=False)
U['grp']=np.where(und,U.record_id,U.address)
# fill missing coords with city median (flagged), so lat/lon models run
for c in ['lat','lon']: U[c]=U[c].fillna(U.groupby('city')[c].transform('median')).fillna(U[c].median())
km=KMeans(12,random_state=RS,n_init=10).fit(U[['lat','lon']]); U['geo_cluster']=km.labels_.astype(str)
U['zip']=U.zip.astype(str)
tracks={'Individual units':U[U.listing_kind=='Individual unit'].reset_index(drop=True),
        'Building floor plans':U[U.listing_kind=='Building floor plan'].reset_index(drop=True)}
feat_prop={'Individual units':(['beds','baths','log_sqft','has_pool','has_ac','has_fireplace','has_spa'],['property_type']),
           'Building floor plans':(['beds'],[])}
def pre(num,cat,scale=False):
    nt=[('imp',SimpleImputer(strategy='median'))]+([('sc',StandardScaler())] if scale else [])
    t=[('n',Pipeline(nt),num)] if num else []
    if cat: t.append(('c',OneHotEncoder(handle_unknown='ignore',min_frequency=5,sparse_output=False),cat))
    return ColumnTransformer(t)
def hgb(): return HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=15,l2_regularization=1.0,random_state=RS)
class Baseline:
    def fit(self,X,y):
        d=X[['beds','city']].copy(); d['y']=y.values
        self.m=d.groupby(['beds','city']).y.agg(['median','size']); self.b=d.groupby('beds').y.median(); self.g=y.median(); return self
    def predict(self,X):
        out=[]
        for b,c in zip(X.beds,X.city):
            if (b,c) in self.m.index and self.m.loc[(b,c),'size']>=3: out.append(self.m.loc[(b,c),'median'])
            else: out.append(self.b.get(b,self.g))
        return np.array(out)
def specs(k):
    pn,pc=feat_prop[k]; loc_c=['city']; 
    S={}
    S['1 Baseline (median beds x city)']=(Baseline(),['beds','city'])
    S['2 Linear (ridge, features+city)']=(Pipeline([('p',pre(pn,pc+loc_c,True)),('m',Ridge(alpha=3))]),pn+pc+loc_c)
    S['3 Random forest (features+city)']=(Pipeline([('p',pre(pn,pc+loc_c)),('m',RandomForestRegressor(400,min_samples_leaf=3,max_features=0.5,n_jobs=-1,random_state=RS))]),pn+pc+loc_c)
    S['4 Gradient boosting (features+city)']=(Pipeline([('p',pre(pn,pc+loc_c)),('m',hgb())]),pn+pc+loc_c)
    S['5 Spatial GBM (+lat/lon, cluster, zip)']=(Pipeline([('p',pre(pn+['lat','lon'],pc+loc_c+['zip','geo_cluster'])),('m',hgb())]),pn+pc+loc_c+['lat','lon','zip','geo_cluster'])
    # ablations
    S['A Property features only (no location)']=(Pipeline([('p',pre(pn,pc)),('m',hgb())]),pn+pc)
    S['B Location only (lat/lon/zip/city)']=(Pipeline([('p',pre(['lat','lon'],['city','zip','geo_cluster'])),('m',hgb())]),['lat','lon','city','zip','geo_cluster'])
    return S
def cv(model,cols,d,n=5):
    oof=np.zeros(len(d)); 
    for tr,te in GroupKFold(n).split(d,groups=d.grp):
        m=__import__('copy').deepcopy(model); m.fit(d.iloc[tr][cols],d.y.iloc[tr]); oof[te]=m.predict(d.iloc[te][cols])
    return oof
rows=[];oofs={}
for k,d in tracks.items():
    for name,(m,cols) in specs(k).items():
        o=cv(m,cols,d); oofs[(k,name)]=o
        y=d.y.values; r=d.rent.values; p=np.exp(o)
        rows.append(dict(track=k,model=name,n=len(d),R2_log=r2_score(y,o),RMSE_log=mean_squared_error(y,o)**.5,MAE_dollars=mean_absolute_error(r,p),MedAPE=np.median(np.abs(r-p)/r),R2_dollars=r2_score(r,p)))
res=pd.DataFrame(rows); print(res.round(3).to_string())
res.to_pickle('model_results.pkl')
# choose best of 1-5 per track by R2_log
best={}
for k in tracks:
    s=res[(res.track==k)&res.model.str[0].isin(list('12345'))].sort_values('R2_log',ascending=False).iloc[0]; best[k]=s.model
print(best)
# value scores
V=[]
for k,d in tracks.items():
    dd=d.copy(); dd['model_used']=best[k]; dd['pred_rent']=np.exp(oofs[(k,best[k])]); 
    # features-only OOF for city premium
    dd['pred_noloc']=np.exp(oofs[(k,'A Property features only (no location)')]); V.append(dd)
V=pd.concat(V,ignore_index=True)
V['score']=(V.pred_rent-V.rent)/V.pred_rent*100
V.to_pickle('values.pkl')
# permutation importance for spatial GBM on IU (held-out folds)
imp=[]
for k,d in tracks.items():
    m0,cols=specs(k)['5 Spatial GBM (+lat/lon, cluster, zip)']
    acc={c:[] for c in cols}
    for tr,te in GroupKFold(5).split(d,groups=d.grp):
        m=__import__('copy').deepcopy(m0).fit(d.iloc[tr][cols],d.y.iloc[tr])
        pi=permutation_importance(m,d.iloc[te][cols],d.y.iloc[te],n_repeats=5,random_state=RS,scoring='r2')
        for c,v in zip(cols,pi.importances_mean): acc[c].append(v)
    for c in cols: imp.append((k,c,np.mean(acc[c])))
imp=pd.DataFrame(imp,columns=['track','feature','r2_drop_when_shuffled']); imp.to_pickle('imp.pkl'); print(imp.sort_values(['track','r2_drop_when_shuffled'],ascending=[True,False]).round(3).to_string())
# OLS-ish premiums (individual units) w/ bootstrap
d=tracks['Individual units'].copy()
d['sqft_per']=d.log_sqft
X=pd.get_dummies(d[['beds','baths','log_sqft','has_pool','has_ac','has_fireplace','has_spa','property_type','city']],columns=['property_type','city'],drop_first=False,dtype=float)
X=X.drop(columns=['property_type_APARTMENT','city_Anaheim'],errors='ignore'); X['log_sqft']=X.log_sqft.fillna(X.log_sqft.median()); X['baths']=X.baths.fillna(X.baths.median())
y=d.y.values; rng=np.random.default_rng(RS); B=300; co=[]
base=LinearRegression().fit(X,y); names=list(X.columns)
for _ in range(B):
    i=rng.integers(0,len(X),len(X)); co.append(LinearRegression().fit(X.iloc[i],y[i]).coef_)
co=np.array(co)
coef=pd.DataFrame({'term':names,'coef_log':base.coef_,'pct_effect':np.exp(base.coef_)-1,'ci_low':np.exp(np.percentile(co,2.5,axis=0))-1,'ci_high':np.exp(np.percentile(co,97.5,axis=0))-1})
coef.to_pickle('coef.pkl'); print(coef[~coef.term.str.startswith('city_')].round(3).to_string())
print('R2 OLS',base.score(X,y))
