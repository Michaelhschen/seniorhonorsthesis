from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]; A=R/'output/analysis'; M=R/'manuscript/generated'
meta=pd.read_csv(A/'source_metadata_as_supplied.csv')
schema=pd.read_csv(R/'config/reproduction_schema.csv').fillna('').set_index('source_variable')
names={'GDP_QoQAnn':'Real GDP growth','CorpCapex_QoQ':'Corporate capital-expenditure growth','CorpFixedCap_QoQ':'Fixed-capital growth (excluding software)','Delinq':'Loan delinquency rate','DelinqChange':'Change in loan delinquency rate','CPI_YoY':'CPI inflation','CorePCE_YoY':'Core PCE inflation','UNRATE':'Unemployment rate','U6UNRATE':'U-6 underutilization rate','ICSA_YoY':'Initial claims growth','HOUST_YoY':'Housing starts growth','DXY_YoY':'Broad U.S. dollar growth','TermSpread_10Y3M':'10-year minus 3-month Treasury spread','HYSpread':'High-yield option-adjusted spread','VIX_log':'Log VIX','M1M2_Ratio':'M1/M2 ratio','ISM_NewOrders':'ISM manufacturing new orders','ISM_PMI':'ISM manufacturing PMI','PhillyFed_BOS':'Philadelphia Fed business outlook'}
units={'qoq_ann':'percent, annualized quarterly growth','qoq_pct':'percent, quarter-on-quarter growth','yoy_pct':'percent, year-on-year growth','diff':'percentage-point change at native frequency','ratio':'unitless ratio','log':'natural log of index'}
level_units={'Delinq':'percent','UNRATE':'percent','U6UNRATE':'percent','TermSpread_10Y3M':'percentage points','HYSpread':'percentage points','ISM_NewOrders':'diffusion index','ISM_PMI':'diffusion index','PhillyFed_BOS':'survey diffusion balance'}
rows=[]
for row in meta.to_dict('records'):
    v=row['variable']; row['economic_name']=names[v]
    row['transformed_panel_units']=units.get(row['transform'],level_units.get(v,'not specified'))
    row['additional_model_transform']=schema.loc[v,'decision'] if v in schema.index else 'drop_default'
    row['source_raw_units']='not recorded in supplied metadata'
    row['source_seasonal_adjustment']='not recorded in supplied metadata'
    if v=='CorpFixedCap_QoQ':row.update(source_raw_units='millions of U.S. dollars',source_seasonal_adjustment='not seasonally adjusted')
    if v=='CorpCapex_QoQ':row.update(source_raw_units='millions of U.S. dollars',source_seasonal_adjustment='seasonally adjusted annual rate')
    if v=='DXY_YoY':row.update(source_raw_units='index January 2006=100',source_seasonal_adjustment='not seasonally adjusted')
    row['definition_verification_url']='https://fred.stlouisfed.org/series/'+str(row['source_id']) if v in ['CorpFixedCap_QoQ','CorpCapex_QoQ','DXY_YoY'] else ''
    row['availability_status']='fixed-lag reconstruction only; release/vintage unverified'
    rows.append(row)
dictionary=pd.DataFrame(rows);dictionary.to_csv(A/'series_dictionary.csv',index=False)
tex=[r'\begin{singlespace}',r'\begin{longtable}{@{}p{0.44\textwidth}p{0.22\textwidth}p{0.25\textwidth}@{}}',r'\caption{Economic variables and additional reproduction transforms}\label{tab:dictionary}\\',r'\toprule Variable & Native frequency & Model treatment \\ \midrule\endfirsthead',r'\toprule Variable & Native frequency & Model treatment \\ \midrule\endhead',r'\bottomrule\endfoot']
for row in rows:
    treat={'keep_level':'Supplied level','keep_difference':'Monthly difference'}.get(row['additional_model_transform'],'Excluded')
    tex.append(f"{row['economic_name']} & {row['frequency'].title()} & {treat}"+r' \\')
tex.extend([r'\end{longtable}',r'\end{singlespace}']);(M/'dictionary.tex').write_text('\n'.join(tex),encoding='utf-8')
lineage=pd.read_csv(R/'package/availability_audit/selected_nowcast_lineage.csv',parse_dates=['origin','issue_date']).set_index(['origin','variable'])
panels={n:pd.read_csv(R/'package/original_run/outputs'/f,index_col=0,parse_dates=True) for n,f in [('FI','full_information.csv'),('RT','realtime_information.csv'),('NC','nowcast_information.csv')]}
ledger=[]
for model,frame in panels.items():
    for origin,values in frame.iterrows():
        for md in rows:
            v=md['variable'];shift=0 if model=='FI' else int(md['fallback_lag_months'])
            source_month=origin.to_period('M')-shift
            ref=str(source_month.asfreq('Q')) if md['frequency']=='quarterly' else str(source_month)
            rec=dict(panel=model,origin=str(origin.date()),variable=v,value=values[v],source_id=md['source_id'],reference_period=ref,source_transform=md['transform'],additional_model_transform=md['additional_model_transform'],fixed_lag_months=shift,issue_date='',vintage_date='',retrieval_timestamp='',availability_verified=False,construction='revised_data_fixed_shift' if model!='FI' else 'revised_data_reference')
            if model=='NC' and (origin,v) in lineage.index:
                item=lineage.loc[(origin,v)]
                rec.update(construction='supplied_history_nowcast',issue_date=str(item.issue_date.date()),reference_period=str(item.target_period),source_id=item.source_url if pd.notna(item.source_url) else item.source_sheet)
                assert abs(values[v]-item.value)<1e-8
            elif model!='FI':
                source_date=source_month.to_timestamp()+pd.offsets.MonthEnd(0)
                if source_date in panels['FI'].index:
                    assert np.isclose(values[v],panels['FI'].loc[source_date,v],equal_nan=True)
            ledger.append(rec)
pd.DataFrame(ledger).to_csv(A/'source_to_panel_ledger.csv',index=False)
print(f'Wrote {len(rows)} dictionary records and {len(ledger)} source-to-panel records; release availability remains unverified')
