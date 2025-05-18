#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy

nlp = spacy.load("en_core_web_sm")

def extract_flat_entities(text):
    doc = nlp(text)
    entities = set()
    
    for token in doc:
        # Извлечение сущностей по синтаксическим правилам
        if token.dep_ in {"nsubj", "nsubjpass", "dobj", "iobj", "pobj", "attr", "advmod", "prep", "prt", "npadvmod"} \
        or token.pos_ == "NOUN":
            span = doc[token.left_edge.i : token.right_edge.i + 1]
            if span.text.strip():
                if len(span.text) > 50:
                    continue
                else:
                    entities.add(span.text)
    
    return list(entities)

# Пример использования
text = "brien et al.23 state ASR value similar fuel cell electrolysis mode operation Fig trend apply decade signiп¬Ѓcantly Ni YSZ high activity oxidation different performance result report SOECs h2 reduction H2O recently report H2 electrode Ni YSZ electrodes25 illustrate manufacturing support cells25 YSZ electrolyte support cells.31,23x electrode microstructure highly inп¬‚uence durable high perform Ni YSZ cermet electrode long term electrode performance fast kinetic involve electrode reaction porous 1980s long term stability 1000 hour Ni YSZ electrode /C24 30 porosity structure ensure fast diffusion H2 electrolyte support tubular soec report DoВЁ nitz electrode support cell disadvantage notice possibly Ni YSZ electrode microstructure relatively coarse19 h2 recycling necessity redox stability issue result large area speciп¬Ѓc polarisation resistance /C24 0.23 importance SOEC test laboratory scale future u cm2 1000 /C14 C19 ASR today state cost competitive SOEC system issue consider art soec report 0.17 u cm2 950 /C14 c cell include ohmic resistance.32 unfortunately initial performance high initial performance satisfy electron microscopy evidence stability report SOEC Ni YSZ H2 electrode Ni YSZ microstructure report DoВЁ nitz et al 2 show iV curve high perform soec Ni YSZ electrode microstructure chord electrolysis iv production method tape cast possible produce curve high perform SOC area speciп¬Ѓc resistance planar soec microstructure illustrate Fig 4A ASR)вЂЎ low 0.27 U cm2 obtain 850 /C14 C p(h2o)/ non test SOEC26,33 mean Ni particle size 1.00 p(H2 Вј 0.5/0.5 cell.25 similar cell test SOFCs /c6 0.05 mm high porosity essentially signiп¬Ѓcant change find 850 /C14 C approximately 25 resistance cell cell conп¬Ѓguration cause process H2 electrode.27 slightly low initial performance single soec obtain OвЂ™Brien et al.24,28 electrolyte support button cell Ni YSZ electrode e.g. 3 long term electrolysis testing single electrolyte support tubular SOEC Ni YSZ H2 electrode report DoВЁ nitz et al.19 test condition 995 /C14 C /C0 0.3 cm C0 2 h2 H2O Вј 1вЃ„2 10 especially sample e6 ss14 ASR value give text reference calculate observed ASR electrolysis mode slightly high ASR chord iV curve linear region fuel cell mode cell"
result = extract_flat_entities(text)
print(result)


# In[ ]:




