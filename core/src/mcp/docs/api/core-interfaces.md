# Core Interfaces API Reference

## Overview

This document provides comprehensive API reference for all core interfaces in the MCP system.

## IMemoryManager Interface

### Description
The IMemoryManager interface defines the contract for memory storage and retrieval operations.

### Methods

#### store(data: Any, key: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str

Stores data in memory with optional key and metadata.

**Parameters:**
- `data` (Any): The data to store
- `key` (Optional[str]): Optional key for the data. If None, a UUID will be generated
- `metadata` (Optional[Dict[str, Any]]): Optional metadata associated with the data

**Returns:**
- `str`: The key used to store the data

**Raises:**
- `MCPMemoryError`: If storage operation fails

**Example:**
```python
memory_manager = BasicMemoryManager()
key = memory_manager.store({"message": "Hello World"}, metadata={"type": "greeting"})
```

#### retrieve(key: str) -> Any

Retrieves data from memory using the provided key.

**Parameters:**
- `key` (str): The key of the data to retrieve

**Returns:**
- `Any`: The stored data

**Raises:**
- `MCPMemoryError`: If key is not found or retrieval fails

**Example:**
```python
data = memory_manager.retrieve(key)
print(data["message"])  # Output: Hello World
```

#### update(key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool

Updates existing data in memory.

**Parameters:**
- `key` (str): The key of the data to update
- `data` (Any): The new data
- `metadata` (Optional[Dict[str, Any]]): Optional metadata to update

**Returns:**
- `bool`: True if update was successful

**Raises:**
- `MCPMemoryError`: If key is not found or update fails

#### delete(key: str) -> bool

Deletes data from memory.

*ul

####ssfcen was sucializatioinitrue if  Tool`: `brns:**
-Retu

**ation configur): Optionalny]]ct[str, A(Optional[Di `config` :**
-eters
**Paramation.
l configurwith optionaes the lobe tializ

Ini) -> bool= Nonestr, Any]] onal[Dict[: Opticonfigalize(initi

#### hods
### Met.
tionsementae implr all lobct foontra base cnes thefie dee interfacILobn
The scriptio## De

#rfaceILobe Inteon.

## nsactirent trahe curk tls bac

RolNone() -> ollback# ron.

###nt transactits the curremione

Comit() -> Nomm c
####action.
nsdatabase traegins a ne

Bion() -> Noansactgin_tr## be

##ults Query rese]`:`List[Tuplurns:**
- s

**Reteteruery paramional q): Optple]nal[Tu (Optio`params`
- teexecury to The SQL query` (str): 
- `queers:**met
**Parae query.
a databasutes Exec]

st[Tuple Line) ->No = ional[Tuple]Optams: r, par(query: ste_queryut

#### exec# Methods

##s.tionperatabase otract for dae cone defines thinterfacseManager baataon
The IDDescripti
### 
centerfaManager Itabase## IDa
ta
daontext exported cstr`: The *
- `Returns:* etc.)

** "yaml",t ("json",ort formatr): Exprmat` (s ID
- `focontextThe _id` (str): 
- `contexters:**

**Parametrmat. foe specified thcontext inExports  -> str

son")t: str = "j formaxt_id: str,ntext(contert_coexpo

#### cessfulate was sucue if upd: Trl` `boons:**
-y

**Returplates to apny]): Upd, Atres` (Dict[s- `updat ID
he context` (str): T `context_id*
-ters:*
**Parametext.
conisting  exUpdates an-> bool

ny]) Dict[str, Aupdates: t_id: str, ntext(contexpdate_co

#### uext data`: The contt[str, Any]- `Dicns:**

**Returtext ID
 The contr):id` (stext_ `coneters:**
-.

**Paramt by IDes contexievAny]

RetrDict[str, ) -> : strext_idtext(cont## get_conD

##context Iique : The un `str`ns:**
-
**Returata
: Context d])[str, AnyDictata` ( `context_d
-rameters:**
**Paded data.
 provithentext from coeates a new str

Cr, Any]) -> Dict[str_data: ontextext(create_contds

#### cMetho

### nagement.ion and mat generatr contexact fo contrhece defines tnterfaanager ihe IContextMion
T### Descripte

nterfacxtManager Ite

## IConfinitionsdekflow  wor List of`:tr, Any]]List[Dict[surns:**
- `er

**Retfiltus ptional stat]): O[strionalOptstatus` (s:**
- `termes.

**Paratured by stailtely f, optionallowsl workf
Lists al, Any]]
ct[strDiList[e) -> l[str] = Nononatatus: Optirkflows(slist_wo# l

###fuccessas suate wupdif ue Tr `bool`: ns:**
-

**Retured")ail, "fted"mple", "conninge.g., "rus (he new statutr): Tstatus` (s- `w ID
he workfloid` (str): Torkflow_
- `wrs:**meteraPa.

** workflowf ahe status oes tdat

Up> boolus: str) -: str, statflow_idtatus(workworkflow_s### update_tion

#kflow defini]`: The wortr, Anyict[s*
- `D**Returns:*

w IDhe workflo` (str): Tworkflow_id- `s:**
eter**Paramby ID.

on definitirkflow Retrieves wo, Any]

> Dict[str -r)kflow_id: stwororkflow( get_w###
```

#w_data)(workfloe_workflowengine.creatrkflow_low_id = wo]
}
workf
    s": {}}"parameterrm_data", nsfotra": " "actionTransform",e": ""nam    {}},
    s": {arametera", "pdatte_idan": "val "actiote","Valida"name":     {: [
    ps" "ste",
   coming datains ocesse: "Prion"  "descriptlow",
  Workfssing Data Proce": "   "name
  {ta =darkflow_ython
wo```pExample:**

**rkflow ID
nique wo`: The u- `streturns:**
ps

**R and steescription,e, daining naminition contefow dorkfl Any]): Wr,(Dict[stlow_data` workfs:**
- `ameter**Parinition.

ovided def prlow from the new workf
Creates atr
) -> sy][str, Ana: Dictrkflow_dat(wokflow# create_wor

###thods.

### Metion execunddefinition aw loworkfact for contrfines the e deacne interfEngirkflowhe IWoion
Tescript### D
 Interface
owEngineorkfl``

## IW")
`data}, Data: {ore: {score}y: {key}, Sc print(f"Kelts:
   re in resudata, scoy, 
for keimit=5)ng", leetigr.search("agerry_man = memon
results*
```pythomple:*
**Exae)
ce_scor relevana,key, datning (les contaif tup List oy, float]]`:ple[str, An`List[Tu
- ****Returns:
rn
ts to retuof resulber numimum  Maximit` (int):ry
- `larch quetr): The sequery` (s
- `meters:***Parastring.

*ry  on queedn memory bas data i forhes

Searcat]] Any, flouple[str,) -> List[T: int = 10 str, limituery:earch(q### s
#n fails
on operatioletirror`: If deMemoryE `MCPaises:**
-*Round

* was not f keyiflse ssful, Faucceetion was s if del`bool`: True:**
- *Returnsdelete

*e data to he key of thy` (str): T**
- `kerameters:*Pa