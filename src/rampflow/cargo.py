from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timedelta
from typing import Iterable,Mapping
from .timeutil import Window,utc
from .models import Decision
from .hazmat import HAZMAT_RULEBOOK,HazmatRulebook

@dataclass(frozen=True)
class ULD:
    uld_id:str
    contour:str
    tare_kg:float
    gross_kg:float
    destination:str
    temperature_band:str=""
    dangerous_groups:frozenset[str]=frozenset()
    priority:int=100

@dataclass(frozen=True)
class CargoPosition:
    position_id:str
    allowed_contours:frozenset[str]
    max_gross_kg:float
    deck:str
    zone:str
    adjacent:frozenset[str]=frozenset()

@dataclass(frozen=True)
class CargoPlan:
    flight_id:str
    assignments:Mapping[str,str]
    total_gross_kg:float
    load_window:Window
    sequence:tuple[str,...]
    # Hazmat rule edition the stowage was planned under.
    hazmat_version:str=""

class CargoManager:
    def __init__(self,positions:Iterable[CargoPosition],rulebook:HazmatRulebook=HAZMAT_RULEBOOK):
        self.positions={p.position_id:p for p in positions}
        self.rulebook=rulebook
        self.plans:dict[str,CargoPlan]={}

    @property
    def hazmat_version(self)->str:
        return self.rulebook.version

    def _compatible(self,uld:ULD,pos:CargoPosition)->bool:
        return uld.contour in pos.allowed_contours and uld.gross_kg<=pos.max_gross_kg

    def _compartment(self,pos:CargoPosition)->tuple[str,str]:
        return (pos.deck,pos.zone)

    def _segregation_conflict(self,uld:ULD,pos:CargoPosition,occupied:Mapping[str,ULD])->tuple[str,str]|None:
        """Incompatible hazmat in the same compartment or an adjacent position."""
        compartment=self._compartment(pos)
        for other_pid,other_uld in occupied.items():
            other_pos=self.positions[other_pid]
            sharing=(self._compartment(other_pos)==compartment or other_pid in pos.adjacent)
            if not sharing:continue
            pair=self.rulebook.conflicting_pair(uld.dangerous_groups,other_uld.dangerous_groups)
            if pair:return pair
        # A single ULD may itself declare two mutually incompatible groups.
        return self.rulebook.conflicting_pair(uld.dangerous_groups,uld.dangerous_groups)

    def plan(self,flight_id:str,ulds:Iterable[ULD],start:datetime,loader_rate_min:int=4)->CargoPlan:
        items=sorted(ulds,key=lambda u:(u.priority,-u.gross_kg,u.uld_id))
        if len({u.uld_id for u in items})!=len(items):
            raise RuntimeError("duplicate ULD in load")
        for u in items:
            if self.rulebook.conflicting_pair(u.dangerous_groups,u.dangerous_groups):
                raise RuntimeError(f"ULD {u.uld_id} declares mutually incompatible dangerous goods")

        def candidates(uld:ULD,by_position:Mapping[str,ULD])->list[str]:
            out=[]
            for pos in self.positions.values():
                if pos.position_id in by_position or not self._compatible(uld,pos):continue
                if self._segregation_conflict(uld,pos,by_position):continue
                out.append(pos)
            out.sort(key=lambda p:(len(p.adjacent),p.deck,p.zone,p.position_id))
            return [p.position_id for p in out]

        # Backtracking search: a greedy placement that dead-ends late must be
        # retried with a different earlier assignment. Either the whole load is
        # stowed in one complete plan or nothing is returned.
        assignments:dict[str,str]={}
        occupied:dict[str,ULD]={}
        def place(i:int)->bool:
            if i==len(items):return True
            uld=items[i]
            for pid in candidates(uld,occupied):
                assignments[uld.uld_id]=pid
                occupied[pid]=uld
                if place(i+1):return True
                del assignments[uld.uld_id]
                del occupied[pid]
            return False

        if not place(0):
            raise RuntimeError("no complete hazmat-segregated stowage for the full load")

        sequence=tuple(u.uld_id for u in sorted(items,key=lambda u:(self.positions[assignments[u.uld_id]].deck,self.positions[assignments[u.uld_id]].zone,u.priority)))
        s=utc(start);duration=max(1,len(sequence)*loader_rate_min)
        window=Window(s,s+timedelta(minutes=duration))
        return CargoPlan(flight_id,dict(assignments),sum(u.gross_kg for u in items),window,sequence,self.rulebook.version)

    def verify(self,plan:CargoPlan,ulds:Mapping[str,ULD])->Decision:
        if plan.hazmat_version and plan.hazmat_version!=self.rulebook.version:
            return Decision(False,"hazmat-edition","plan stowed under a different hazmat edition",{"plan_edition":plan.hazmat_version,"current_edition":self.rulebook.version})
        seen=set()
        for uid,pid in plan.assignments.items():
            if uid not in ulds or pid not in self.positions:return Decision(False,"reference","cargo plan references unknown item")
            if pid in seen:return Decision(False,"position","cargo position reused")
            seen.add(pid)
            if not self._compatible(ulds[uid],self.positions[pid]):return Decision(False,"compatibility","ULD incompatible with assigned position",{"uld":uid,"position":pid})
        assigned=[(uid,pid) for uid,pid in plan.assignments.items()]
        for i,(uid,pid) in enumerate(assigned):
            pos=self.positions[pid]
            if self.rulebook.conflicting_pair(ulds[uid].dangerous_groups,ulds[uid].dangerous_groups):
                return Decision(False,"hazmat-separation","ULD carries mutually incompatible dangerous goods",{"uld":uid})
            for other_uid,other_pid in assigned[i+1:]:
                other=self.positions[other_pid]
                sharing=(self._compartment(other)==self._compartment(pos) or other_pid in pos.adjacent)
                if not sharing:continue
                pair=self.rulebook.conflicting_pair(ulds[uid].dangerous_groups,ulds[other_uid].dangerous_groups)
                if pair:
                    return Decision(False,"hazmat-separation","incompatible dangerous cargo share compartment or adjacency",{"uld":uid,"other":other_uid,"pair":pair})
        return Decision(True,"ok","cargo plan valid",{"gross_kg":plan.total_gross_kg,"hazmat_version":self.rulebook.version})

    def commit(self,plan:CargoPlan,ulds:Mapping[str,ULD])->Decision:
        d=self.verify(plan,ulds)
        if not d.accepted:return d
        self.plans[plan.flight_id]=plan
        return Decision(True,"ok","cargo plan committed",{"sequence":plan.sequence,"hazmat_version":self.rulebook.version})
