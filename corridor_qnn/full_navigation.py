"""Privileged static Doom-format IWAD geometry planner, shared by all policies.

Rasterized collision approximation, not an engine pathfinder. Doors are considered
potentially traversable. Locks, lifts, teleports and scripted specials are not solved.
No map-specific coordinates or action sequences are stored.
"""
from collections import deque
from pathlib import Path
import struct
import numpy as np
from PIL import Image, ImageDraw


def map_name(value, secret=False):
    value = str(value).upper()
    if len(value) == 4 and value[0] == 'E' and value[2] == 'M' and value[1] in '1234' and value[3] in '123456789':
        return value
    if len(value) != 5 or not value.startswith('MAP') or not value[3:].isdigit():
        raise ValueError('Map must be MAP01 through MAP30 (MAP31/32 with --secret-maps) or ExMy (E1M1-E4M9)')
    if not 1 <= int(value[3:]) <= (32 if secret else 30):
        raise ValueError('Map outside supported range')
    return value


def _map_lumps(path, name):
    data = Path(path).read_bytes()
    magic, count, offset = struct.unpack_from('<4sii', data)
    if magic not in (b'IWAD', b'PWAD'):
        raise ValueError('Invalid WAD header')
    lumps = [struct.unpack_from('<ii8s', data, offset+16*i) for i in range(count)]
    names = [n.rstrip(b'\0').decode('ascii') for _, _, n in lumps]
    if name not in names:
        raise ValueError(f'{name} absent from {path}')
    start = names.index(name)
    found = {}
    for off, size, n in lumps[start+1:start+12]:
        found[n.rstrip(b'\0').decode('ascii')] = data[off:off+size]
    if 'BEHAVIOR' in found or 'TEXTMAP' in found:
        raise ValueError('Planner currently supports classic Doom-format maps only')
    return found


def read_things(path, name):
    """(x, y, angle, type, flags) records; empty when the map has no THINGS lump."""
    found = _map_lumps(path, name)
    return list(struct.iter_unpack('<hhhhh', found.get('THINGS', b'')))


def read_map(path, name):
    found = _map_lumps(path, name)
    def records(key, fmt):
        return list(struct.iter_unpack(fmt, found[key]))
    verts = records('VERTEXES', '<hh')
    lines = records('LINEDEFS', '<HHHHHHH')
    sides = records('SIDEDEFS', '<hh8s8s8sH')
    sectors = records('SECTORS', '<hh8s8shhh')
    return verts, lines, sides, sectors


class LegacyNavigator:
    cell = 32.0

    def __init__(self, path, name):
        verts, lines, sides, sectors = read_map(path, name)
        xy = np.asarray(verts)
        self.origin = xy.min(0)-96
        size = np.ceil((xy.max(0)-self.origin+96)/self.cell).astype(int)+1
        if np.prod(size) > 2_000_000:
            raise ValueError('Map too large for bounded navigation grid')
        image = Image.new('L', tuple(size), 0)
        draw = ImageDraw.Draw(image)
        self.exits = []
        self.interactions = []
        for v1, v2, flags, special, tag, right, left in lines:
            p, q = xy[v1], xy[v2]
            if special in (11, 51, 52, 124, 197, 198):
                self.exits.append((p+q)/2)
            if special:
                self.interactions.append((p+q)/2)
            solid = left == 65535 or right == 65535 or bool(flags & 1)
            if not solid:
                a, b = sectors[sides[right][-1]], sectors[sides[left][-1]]
                door = a[0] == a[1] or b[0] == b[1]
                solid = not door and (abs(a[0]-b[0]) > 24 or min(a[1],b[1])-max(a[0],b[0]) < 56)
            if solid:
                a, b = (p-self.origin)/self.cell, (q-self.origin)/self.cell
                draw.line([tuple(a), tuple(b)], fill=1, width=1)
        self.blocked = np.asarray(image).astype(bool)
        self.reset()

    def reset(self):
        self.visits = np.zeros(self.blocked.shape, dtype=np.int32)
        self.path = []
        self.reachable = 1
        self.calls = 0

    def grid(self, xy):
        x, y = np.rint((np.asarray(xy)-self.origin)/self.cell).astype(int)
        return int(x), int(y)

    def world(self, node):
        return np.asarray(node)*self.cell+self.origin

    def free(self, node):
        x,y = node
        return 0 <= y < self.blocked.shape[0] and 0 <= x < self.blocked.shape[1] and not self.blocked[y,x]

    def route(self, start):
        queue = deque([start]); parent = {start: None}; dist = {start: 0}
        while queue:
            x,y = queue.popleft()
            for node in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if node not in parent and self.free(node):
                    parent[node]=(x,y); dist[node]=dist[(x,y)]+1; queue.append(node)
        self.reachable = len(parent)
        # Approach the nearest reachable cell to an actual exit linedef.
        if self.exits:
            goal = min(parent, key=lambda n: min(np.linalg.norm(self.world(n)-e) for e in self.exits))
            if self.visits[goal[1],goal[0]] > 8:
                goal = max(parent, key=lambda n: dist[n]/(1+self.visits[n[1],n[0]]*10))
        else:
            goal = max(parent, key=lambda n: dist[n]/(1+self.visits[n[1],n[0]]*10))
        path=[]
        while goal != start:
            path.append(goal); goal=parent[goal]
        return path[::-1]

    def update(self, xy):
        node = self.grid(xy)
        if 0 <= node[1] < self.visits.shape[0] and 0 <= node[0] < self.visits.shape[1]:
            self.visits[node[1],node[0]] += 1
        if not self.path or self.calls % 32 == 0 or np.linalg.norm(self.world(self.path[0])-xy)>160:
            self.path = self.route(node)
        self.calls += 1
        while len(self.path)>1 and np.linalg.norm(self.world(self.path[0])-xy)<45:
            self.path.pop(0)
        target = self.world(self.path[0]) if self.path else np.asarray(xy)
        near_use = min((np.linalg.norm(p-xy) for p in self.interactions), default=1e6)
        return target, near_use


class Navigator(LegacyNavigator):
    """Player-clearance grid, front-side use goals and local collision memory.

    Static IWAD information is privileged, as in the original planner. Closed
    doors are optimistic portals. Key-locked doors stay closed until the player
    passes over the matching key. Sectors moved by tagged triggers (lifts,
    lowering floors, remote doors) use live engine heights; when neither exit
    nor a missing key is reachable, unused switches, walk triggers and
    teleporters become subgoals. No map coordinates or recorded action sequences.
    """
    cell = 16.0
    radius = 18.0
    use_specials = {1, 11, 26, 27, 28, 29, 31, 32, 33, 34, 42, 50, 51,
                    61, 63, 103, 111, 112, 113, 114, 115, 116, 117, 118,
                    133, 134, 135, 136, 137}
    locks = {26:'blue', 32:'blue', 99:'blue', 133:'blue', 27:'yellow', 34:'yellow',
             136:'yellow', 137:'yellow', 28:'red', 33:'red', 134:'red', 135:'red'}
    key_things = {5:'blue', 40:'blue', 6:'yellow', 39:'yellow', 13:'red', 38:'red'}
    exit_specials = {11, 51, 52, 124, 197, 198}
    manual_doors = {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}
    switch_specials = {7, 9, 14, 15, 18, 20, 21, 23, 29, 41, 42, 43, 45, 49, 50, 55,
                       60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 99, 101, 102,
                       103, 111, 112, 113, 114, 115, 116, 122, 123, 127, 131, 132,
                       133, 134, 135, 136, 137, 138, 139, 140}
    teleport_specials = {39, 97}
    lift_specials = {10, 21, 62, 88, 120, 121, 122, 123}
    health_things = {2011: 10, 2012: 25, 2013: 100}  # stimpack, medikit, soulsphere
    damaging_specials = {4, 5, 7, 11, 16}  # nukage, slime, and strobe/damage floors
    # Solid decorations (barrels, lamps, columns, trees, impaled bodies) block movement.
    solid_things = {2035, 2028, 70, 85, 86, 48, 30, 31, 32, 33, 35, 36, 37, 41, 42, 43, 44,
                    45, 46, 47, 54, 55, 56, 57, 25, 26, 27, 28, 29, 49, 50, 51, 52, 53,
                    73, 74, 75, 76, 77, 78}
    # Gun-activated and monster-only lines cannot be used by walking or USE.
    unusable_specials = {24, 46, 47, 125, 126}
    once_specials = {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                     22, 23, 25, 29, 30, 35, 36, 37, 38, 39, 40, 41, 44, 49, 50, 53,
                     55, 56, 57, 58, 59, 71, 100, 101, 102, 103, 104, 108, 109, 110,
                     111, 112, 113, 119, 121, 122, 127, 130, 131, 133, 135, 137, 140, 141}

    def __init__(self, path, name, version='improved'):
        # 'improved2' behaviour is opt-in so checkpoints trained on 'improved' see the
        # same waypoints they were trained on.
        self.version = version
        verts, lines, sides, sectors = read_map(path, name)
        xy = np.asarray(verts, dtype=float)
        self.origin = xy.min(0)-96
        size = np.ceil((xy.max(0)-self.origin+96)/self.cell).astype(int)+1
        if np.prod(size)>2_000_000: raise ValueError('Map too large for bounded navigation grid')
        self.blocked = np.zeros(tuple(size[::-1]), dtype=bool)
        self.exits=[]; self.interactions=[]; self.use_lines=[]; self.walls=[]
        self.exit_approaches=[]; self.locked={}; self.triggers=[]; self.dynamic_lines=[]; self.dynamic_geometry=[]
        self.static_heights=[list(sec[:2]) for sec in sectors]
        targeted={tag for _,_,_,special,tag,_,_ in lines if special and tag}
        self.dynamic_sectors={i for i,sec in enumerate(sectors) if sec[6] in targeted}
        lift_tags={tag for _,_,_,special,tag,_,_ in lines if special in self.lift_specials and tag}
        self.lift_sectors=[i for i,sec in enumerate(sectors) if sec[6] in lift_tags]
        things=read_things(path,name)
        self.keys=[(self.key_things[t],np.array([x,y],dtype=float)) for x,y,_,t,_ in things if t in self.key_things]
        # Flag 1 marks things present on skills 1-2, the default training difficulty.
        self.health_items=[np.array([x,y],dtype=float) for x,y,_,t,flags in things
                           if t in self.health_things and flags&1 and not flags&16]
        # Doom's actor footprint can span a narrow depression. Treating each
        # bank as a separate impassable step incorrectly disconnects bridges.
        bounds={}; neighbors={}; ledges=[]
        for v1,v2,flags,special,tag,right,left in lines:
            ids=[sides[z][-1] for z in (right,left) if z!=65535]
            for sid in ids:
                bounds.setdefault(sid,[]).extend((xy[v1],xy[v2]))
                neighbors.setdefault(sid,set()).update(set(ids)-{sid})
        # Lift platforms are rectangular in practice; a shrunken bounding box locates the rider.
        self.lift_boxes={i:(np.min(bounds[i],0)+8,np.max(bounds[i],0)-8) for i in self.lift_sectors
                         if i in bounds and min(np.ptp(bounds[i],0))>=48}
        self.spanned_sectors={}
        self.lift_centers={}
        for sid,points in bounds.items():
            extent=np.ptp(points,axis=0)
            adjacent=[sectors[i][0] for i in neighbors[sid]]
            if adjacent and min(extent)<2*16 and max(adjacent)-min(adjacent)<=24 and min(adjacent)-sectors[sid][0]>24:
                self.spanned_sectors[sid]=min(adjacent)
        for v1,v2,flags,special,tag,right,left in lines:
            p,q=xy[v1],xy[v2]; midpoint=(p+q)/2
            normal=np.array([q[1]-p[1],p[0]-q[0]])/max(np.linalg.norm(q-p),1)
            if special in self.exit_specials:
                self.exits.append(midpoint)
                # Doom switch activation uses the front (right) side of a linedef.
                self.exit_approaches.append((midpoint+normal*40,midpoint,normal))
            switch=special in self.switch_specials and special not in self.exit_specials
            if special in self.use_specials or switch:
                self.interactions.append(midpoint); self.use_lines.append((p,q,midpoint))
            if tag and special and special not in self.exit_specials|self.manual_doors|self.unusable_specials:
                kind='switch' if switch else 'teleport' if special in self.teleport_specials else 'walk'
                self.triggers.append(dict(kind=kind,p=p,q=q,mid=midpoint,normal=normal,
                                          once=special in self.once_specials,
                                          targets={i for i,sec in enumerate(sectors) if sec[6]==tag}))
            if special in self.locks:
                grid=self.locked.setdefault(self.locks[special],np.zeros_like(self.blocked))
                self._raster_segment(p,q,self.radius,grid)
            solid=left==65535 or right==65535 or bool(flags&1)
            if not solid and {sides[right][-1],sides[left][-1]}&self.dynamic_sectors:
                # Passability is decided from live heights in _update_dynamic.
                self.dynamic_lines.append((sides[right][-1],sides[left][-1],special in self.manual_doors,
                                           self._segment_cells(p,q,self.radius)))
                self.dynamic_geometry.append((p,q,normal))
                continue
            if not solid:
                ra,lb=sides[right][-1],sides[left][-1]
                a,b=list(sectors[ra]),list(sectors[lb])
                a[0]=self.spanned_sectors.get(ra,a[0]); b[0]=self.spanned_sectors.get(lb,b[0])
                door=a[0]==a[1] or b[0]==b[1]
                solid=not door and (abs(a[0]-b[0])>24 or min(a[1],b[1])-max(a[0],b[0])<56)
                # A ledge more than 24 units high can still be stepped off when headroom allows.
                if solid and abs(a[0]-b[0])>24 and min(a[1],b[1])-max(a[0],b[0])>=56:
                    ledges.append((p,q,normal if a[0]>b[0] else -normal))
            if solid:
                self.walls.append((p,q))
                self._raster_segment(p,q,self.radius)
        for x,y,_,t,flags in things:
            # Skip multiplayer-only things (flag 16); skill filtering is not applied.
            if t in self.solid_things and not flags&16:
                c=np.array([x,y],dtype=float); self._raster_segment(c,c,16+self.radius)
        for sid,(lo,hi) in self.lift_boxes.items():
            # Stand inside the ring of a lift's walk triggers, so stepping off does not lower it under us.
            ring=[t['mid'] for t in self.triggers if t['kind']=='walk' and sid in t['targets']]
            if self.version=='improved2':
                # Lifts sharing a tag share their trigger lines: keep only this lift's ring.
                ring=[m for m in ring if np.all(m>=lo-96) and np.all(m<=hi+96)]
            centre=np.mean(ring,0) if len(ring)>=3 else (lo+hi)/2
            if self.version=='improved2' and not (np.all(centre>=lo) and np.all(centre<=hi)): centre=(lo+hi)/2
            self.lift_centers[sid]=centre
        # Cells inside damaging sectors (even-odd test against each sector's edges).
        self.hazard=np.zeros_like(self.blocked)
        edges={}
        for v1,v2,flags,special,tag,right,left in lines:
            for side in (right,left):
                if side!=65535: edges.setdefault(sides[side][-1],[]).append((xy[v1],xy[v2]))
        for sid,sec in enumerate(sectors):
            if sec[5] not in self.damaging_specials or sid not in edges: continue
            seg=np.array(edges[sid]); lo=seg.reshape(-1,2).min(0); hi=seg.reshape(-1,2).max(0)
            (x0,y0),(x1,y1)=self.grid(lo),self.grid(hi)
            xx,yy=np.meshgrid(np.arange(max(x0,0),min(x1,self.blocked.shape[1]-1)+1),
                              np.arange(max(y0,0),min(y1,self.blocked.shape[0]-1)+1))
            px=xx*self.cell+self.origin[0]; py=yy*self.cell+self.origin[1]; inside=np.zeros(xx.shape,bool)
            for (ax,ay),(bx,by) in seg:
                if ay==by: continue
                crosses=((ay>py)!=(by>py))&(px<ax+(py-ay)*(bx-ax)/(by-ay))
                inside^=crosses
            self.hazard[yy[inside],xx[inside]]=True
        self.avoid_hazard=False
        # Cell pairs across moving-sector lines; which way is a drop depends on live heights.
        self.dynamic_pairs=[]
        for p,q,normal in self.dynamic_geometry:
            pairs=[]; length=np.linalg.norm(q-p)
            for t in np.arange(8,length-7,8)/max(length,1):
                point=p+t*(q-p)
                right,left=self.grid(point+normal*(self.radius+10)),self.grid(point-normal*(self.radius+10))
                if LegacyNavigator.free(self,right) and LegacyNavigator.free(self,left): pairs.append((right,left))
            self.dynamic_pairs.append(pairs)
        self.dyn_drops={}
        # One-way drop edges from the high side of each ledge to the low side.
        self.drops={}
        for p,q,down in ledges:
            length=np.linalg.norm(q-p)
            for t in np.arange(8,length-7,8)/max(length,1):
                point=p+t*(q-p)
                high,low=self.grid(point+down*(self.radius+10)),self.grid(point-down*(self.radius+10))
                if LegacyNavigator.free(self,high) and LegacyNavigator.free(self,low):
                    self.drops.setdefault(high,set()).add(low)
        self.reset()

    def _raster_segment(self,p,q,radius,grid=None):
        grid=self.blocked if grid is None else grid
        grid.flat[self._segment_cells(p,q,radius)]=True

    def _segment_cells(self,p,q,radius):
        """Flat grid indices within radius of segment pq."""
        lo=np.maximum(0,np.floor((np.minimum(p,q)-radius-self.origin)/self.cell).astype(int))
        hi=np.minimum(np.array(self.blocked.shape[::-1])-1,np.ceil((np.maximum(p,q)+radius-self.origin)/self.cell).astype(int))
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
        pts=np.stack((xx,yy),axis=-1)*self.cell+self.origin
        delta=q-p; t=np.clip(np.sum((pts-p)*delta,axis=-1)/max(np.dot(delta,delta),1),0,1)
        hit=np.linalg.norm(pts-(p+t[...,None]*delta),axis=-1)<radius
        return np.ravel_multi_index((yy[hit],xx[hit]),self.blocked.shape)

    def reset(self):
        super().reset()
        self.recent=deque(maxlen=12); self.temporary={}; self.goal=None
        self.mode='route'; self.replans=0; self.collision_events=0; self.use_wait=0
        self.held=set(); self._update_locks()
        self.heights=[h[:] for h in self.static_heights]; self.dynamic_state=None
        self.dyn_blocked=np.zeros_like(self.blocked); self.dropping=False; self._update_dynamic()
        self.last_xy=None
        for t in self.triggers: t['until']=0; t['pending']=None
        self.trigger_goal=None; self.trigger_since=0; self.best_reach=0; self.boarded={}; self.used_targets=set()
        self.board=None  # (lift sector, deadline) after pressing a nearby lift switch
        self.known_reach=set(); self.taken=set(); self.health=100; self.advance_goal=None
        self.gain_cache={}; self.health_since=0

    def _update_dynamic(self):
        """Rasterize lines touching moving sectors from current heights; True if changed."""
        h=self.heights; state=[]
        for a,b,manual,_ in self.dynamic_lines:
            fa,ca=h[a]; fb,cb=h[b]
            door=manual and (fa==ca or fb==cb)
            state.append(not door and (abs(fa-fb)>24 or min(ca,cb)-max(fa,fb)<56))
        if state==self.dynamic_state: return False
        self.dynamic_state=state; self.dyn_blocked[:]=False
        cells=[line[3] for line,solid in zip(self.dynamic_lines,state) if solid]
        if cells: self.dyn_blocked.flat[np.concatenate(cells)]=True
        self.dyn_drops={}
        for (a,b,_,_),solid,pairs in zip(self.dynamic_lines,state,getattr(self,'dynamic_pairs',())):
            (fa,ca),(fb,cb)=h[a],h[b]
            if solid and abs(fa-fb)>24 and min(ca,cb)-max(fa,fb)>=56:
                for right,left in pairs:
                    high,low=(right,left) if fa>fb else (left,right)
                    self.dyn_drops.setdefault(high,set()).add(low)
        return True

    def _update_locks(self):
        self.lock_mask=np.zeros_like(self.blocked)
        for color,grid in self.locked.items():
            if color not in self.held: self.lock_mask|=grid

    def free(self,node):
        return (super().free(node) and not self.lock_mask[node[1],node[0]]
                and not (self.avoid_hazard and self.hazard[node[1],node[0]])
                and (self.dropping or not self.dyn_blocked[node[1],node[0]])
                and self.temporary.get(node,0)<=self.calls)

    def clear(self,a,b):
        # Sample at half-cell spacing; also used to avoid cutting obstacle corners.
        n=max(2,int(np.linalg.norm(np.asarray(b)-a)/(self.cell/2))+1)
        return all(self.free(self.grid(p)) for p in np.linspace(a,b,n))

    hazard_cost = 10  # one damaging cell weighs as much as ten safe ones

    def _search(self,start):
        if self.hazard.any():
            # Dijkstra: damaging floors are crossed only where needed, by the shortest stretch.
            import heapq
            heap=[(0,start)]; parent={start:None}; dist={start:0}
            while heap:
                d,(x,y)=heapq.heappop(heap)
                if d>dist[(x,y)]: continue
                drops=[n for n in (*self.drops.get((x,y),()),*self.dyn_drops.get((x,y),())) if not self.hazard[n[1],n[0]]]
                for node in ((x+1,y),(x-1,y),(x,y+1),(x,y-1),*drops):
                    # Never drop into a damaging pit: there may be no way back up.
                    if not self.free(node): continue
                    nd=d+(self.hazard_cost if self.hazard[node[1],node[0]] else 1)
                    if nd<dist.get(node,np.inf):
                        dist[node]=nd; parent[node]=(x,y); heapq.heappush(heap,(nd,node))
            return parent,dist
        queue=deque([start]); parent={start:None}; dist={start:0}
        while queue:
            x,y=queue.popleft()
            for node in ((x+1,y),(x-1,y),(x,y+1),(x,y-1),*self.drops.get((x,y),()),*self.dyn_drops.get((x,y),())):
                if node not in parent and self.free(node):
                    parent[node]=(x,y); dist[node]=dist[(x,y)]+1; queue.append(node)
        return parent,dist

    def _trigger_gain(self,t,start,parent):
        """Value of opening t's target sectors: a key or exit made reachable, else new ground."""
        key=(id(t),frozenset(self.held),tuple(self.dynamic_state),len(parent))
        if key in self.gain_cache: return self.gain_cache[key]
        saved=self.dyn_blocked; self.dyn_blocked=np.zeros_like(saved)
        cells=[c for (a,b,_,c),solid in zip(self.dynamic_lines,self.dynamic_state)
               if solid and not {a,b}&t['targets']]
        if cells: self.dyn_blocked.flat[np.concatenate(cells)]=True
        opened,_=self._search(start); self.dyn_blocked=saved
        new=[n for n in opened if n not in parent]
        gain=sum(1 if n not in self.known_reach else .1 for n in new)
        pts=np.array(list(opened))*self.cell+self.origin
        before=np.array(list(parent))*self.cell+self.origin
        for color,pos in self.keys:
            if color not in self.held and (np.linalg.norm(pts-pos,axis=1)<56).any() \
               and not (np.linalg.norm(before-pos,axis=1)<56).any(): gain+=100_000
        for approach,mid,normal in self.exit_approaches:
            if (np.linalg.norm(pts-approach,axis=1)<32).any(): gain+=1_000_000
        self.gain_cache[key]=gain
        return gain

    def route(self,start):
        if not self.free(start):
            # The clearance raster can mark a narrow channel (or a temporary block) as
            # occupied: plan from the nearest free cell, and walk to it first.
            good=[]
            for r in ((3,8) if self.version=='improved2' else (3,)):
                good=[(start[0]+x,start[1]+y) for x in range(-r,r+1) for y in range(-r,r+1)
                      if self.free((start[0]+x,start[1]+y))]
                if good: break
            if not good: return []
            nearest=min(good,key=lambda n:np.linalg.norm(np.asarray(n)-start))
            if self.version!='improved2':
                start=nearest
            else:
                path=self.route(nearest)
                return [nearest]+path if nearest!=start else path
        parent,dist=self._search(start)
        if len(parent)<30 and self.dyn_blocked.any():
            # Stranded on a raised platform: stepping down is allowed although stepping up is not.
            self.dropping=True; parent,dist=self._search(start); self.dropping=False
            goal=max(parent,key=lambda n:min(dist[n],40)/(1+self.visits[n[1],n[0]]*8))
            self.goal=None; self.mode='drop'; self.reachable=len(parent); self.replans+=1
            path=[]
            while goal!=start:
                path.append(goal); goal=parent[goal]
            return path[::-1]
        self.reachable=len(parent); self.replans+=1
        nodes=list(parent); points=np.array(nodes)*self.cell+self.origin
        choices=[]; mode='exit'
        for approach,mid,normal in self.exit_approaches:
            distance=np.linalg.norm(points-approach,axis=1)
            front=(points-mid)@normal>20
            for i in np.flatnonzero(front & (distance<32)):
                choices.append((dist[nodes[i]]+distance[i]/self.cell,nodes[i],mid))
        # Top up before pushing on: fixed ambushes (monster closets, key traps) kill a
        # weakened player. 64 seeds: 94% completion versus 77% with a threshold of 70.
        if not choices and self.health<90:
            # Low health: detour to the nearest reachable health item not yet taken.
            for k,pos in enumerate(self.health_items):
                if k in self.taken: continue
                for i in np.flatnonzero(np.linalg.norm(points-pos,axis=1)<48):
                    # The more hurt, the longer the detour worth taking.
                    limit=60 if self.health>=70 else np.inf
                    if dist[nodes[i]]<limit: choices.append((dist[nodes[i]],nodes[i],pos,k))
            mode='health'
        if not choices:
            # Exit sealed by a locked door: fetch a reachable key not yet held.
            for color,pos in self.keys:
                if color in self.held: continue
                # Items are touched when bounding boxes overlap, even on a ledge up to body height.
                distance=np.linalg.norm(points-pos,axis=1)
                for i in np.flatnonzero(distance<56):
                    choices.append((dist[nodes[i]],nodes[i],pos))
            mode='key'
        if not choices and self.known_reach:
            # A trigger or key just opened new ground (e.g. a lowered lift): go there.
            # Damaging floors are never a destination in themselves.
            fresh=[n for n in parent if n not in self.known_reach and not self.hazard[n[1],n[0]]]
            if len(fresh)>=4: self.advance_goal=min(fresh,key=lambda n:abs(dist[n]-24))
            if self.advance_goal in parent and dist[self.advance_goal]>2:
                choices=[(dist[self.advance_goal],self.advance_goal,None)]
            else: self.advance_goal=None
            mode='advance'
        self.known_reach|=set(parent)
        if not choices:
            # Sealed area: activate the most useful reachable trigger, then replan on new heights.
            # One-step lookahead: open the trigger's target sectors and measure what becomes reachable.
            # Trapped on a damaging floor with no safe cell: retry escape triggers regardless of cooldown.
            trapped=(self.version=='improved2' and self.hazard[start[1],start[0]]
                     and all(self.hazard[n[1],n[0]] for n in parent))
            for t in self.triggers:
                if t['until']>self.calls and not (trapped and t['pending'] is None): continue
                if t['kind']=='switch':
                    near=(np.linalg.norm(points-(t['mid']+t['normal']*40),axis=1)<32)&((points-t['mid'])@t['normal']>20)
                    goals=[t['mid']]*len(points)
                else:
                    near=np.linalg.norm(points-t['mid'],axis=1)<24
                    side=np.sign((points-t['mid'])@t['normal'])
                    if t['kind']=='teleport': near&=side>0
                    # Doom fires walk lines when the player's centre crosses: aim past the line.
                    goals=[t['mid']-(sd or 1)*t['normal']*24 for sd in side]
                hits=np.flatnonzero(near)
                if not len(hits): continue
                i=min(hits,key=lambda i:dist[nodes[i]])
                gain=self._trigger_gain(t,start,parent)
                choices.append((-gain/(1+dist[nodes[i]]/100),nodes[i],goals[i],id(t)))
            mode='trigger'
        if mode=='trigger' and self.reachable>self.best_reach+50:
            # New territory: repeatable triggers may matter again.
            self.best_reach=self.reachable
            for t in self.triggers:
                if not t['once']: t['until']=0
        if choices:
            # Commit to the current trigger while it stays reachable, avoiding back-and-forth.
            kept=[c for c in choices if mode=='trigger' and c[3]==self.trigger_goal]
            best=min(kept or choices,key=lambda x:x[0]); goal,self.goal=best[1],best[2]; self.mode=mode
            chosen=best[3] if mode=='trigger' else None
            if chosen!=self.trigger_goal: self.trigger_goal=chosen; self.trigger_since=self.calls
        else:
            goal=start; self.goal=None
        if goal==start and self.goal is None:
            # Nothing to aim at (or already there): favor unvisited reachable regions,
            # retaining travel-cost pressure. Never return an empty path without a goal
            # point, which would leave the waypoint on the player and spin it in place.
            if self.version=='improved2' and self.hazard.any() and not self.hazard[start[1],start[0]]:
                # Exploring never justifies damage: search again with damaging floors excluded.
                self.avoid_hazard=True; parent,dist=self._search(start); self.avoid_hazard=False
            if self.version=='improved2' and self.hazard[start[1],start[0]]:
                # Standing on a damaging floor with nothing to do: step off it first.
                goal=min((n for n in parent if not self.hazard[n[1],n[0]]),default=start,key=lambda n:dist[n])
            else:
                goal=max((n for n in parent if n!=start and not self.hazard[n[1],n[0]]),default=start,
                         key=lambda n:min(dist[n],40)/(1+self.visits[n[1],n[0]]*8))
            self.mode='explore'; self.advance_goal=None
        path=[]
        while goal!=start:
            path.append(goal); goal=parent[goal]
        return path[::-1]

    def update(self,xy,yaw=None,last_action=-1,enemy_visible=False,heights=None,health=None):
        xy=np.asarray(xy); node=self.grid(xy)
        self.calls+=1
        if health is not None:
            if (health<90)!=(self.health<90): self.path=[]
            self.health=health
        for k,pos in enumerate(self.health_items):
            if k in self.taken: continue
            if np.max(np.abs(pos-xy))<36:
                self.taken.add(k)
                if self.mode=='health': self.path=[]
            elif self.mode=='health' and self.goal is not None and np.array_equal(self.goal,pos):
                # Out of reach in practice (e.g. on a high ledge): give up after a while nearby.
                if np.linalg.norm(pos-xy)>96: self.health_since=self.calls
                elif self.calls-self.health_since>150: self.taken.add(k); self.path=[]
        if heights is not None:
            self.heights=heights
            if self._update_dynamic(): self.path=[]
        for sid,(lo,hi) in self.lift_boxes.items():
            # Boarded from below: ride up, since the upper level only connects on arrival.
            # Boarded at the top: the lift is carrying us down, so step off at the bottom.
            if not (np.all(xy>=lo) and np.all(xy<=hi)):
                self.boarded.pop(sid,None); continue
            lowered=self.heights[sid][0]<self.static_heights[sid][0]
            self.boarded.setdefault(sid,'below' if lowered else 'top')
            if not lowered: self.boarded[sid]='top'
            elif self.boarded[sid]=='below':
                self.mode='ride'; self.path=[]; self.board=None
                return self.lift_centers[sid],1e6
        if self.board is not None:
            sid,deadline=self.board
            if self.calls<deadline and self.heights[sid][0]<self.static_heights[sid][0]:
                # The lift is down for only a few seconds: walk straight onto it.
                self.mode='board'; self.path=[]
                return self.lift_centers[sid],1e6
            if self.calls>=deadline: self.board=None
        for t in self.triggers:
            if t['pending'] is not None:
                # A press only counts once a target sector actually moves; otherwise retry later.
                deadline,base=t['pending']
                if [tuple(self.heights[i]) for i in sorted(t['targets'])]!=base: t['pending']=None
                elif self.calls>=deadline: t['pending']=None; t['until']=0; self.path=[]
            if t['until']>self.calls: continue
            delta=t['q']-t['p']; s=np.clip(np.dot(xy-t['p'],delta)/max(np.dot(delta,delta),1),0,1)
            d=np.linalg.norm(xy-(t['p']+s*delta))
            if t['kind']=='switch':
                # USE reaches the first line along the view ray: require the front side and facing it.
                facing=yaw is not None and np.dot([np.cos(np.radians(yaw)),np.sin(np.radians(yaw))],-t['normal'])>.7
                fired=d<64 and last_action in (12,13) and (xy-t['p'])@t['normal']>0 and facing
            elif self.last_xy is None: fired=False
            elif t['kind']=='teleport':
                fired=np.linalg.norm(xy-self.last_xy)>96 and np.linalg.norm(self.last_xy-t['mid'])<64
            else:
                before=(self.last_xy-t['p'])@t['normal']; after=(xy-t['p'])@t['normal']
                fired=before*after<=0 and before!=after and 0<s<1
            if fired:
                t['until']=np.inf if t['once'] else self.calls+300; self.path=[]; self.used_targets|=t['targets']
                if t['kind']!='teleport':
                    t['pending']=(self.calls+60,[tuple(self.heights[i]) for i in sorted(t['targets'])])
                if self.version=='improved2':
                    near_lift=[sid for sid in t['targets'] if sid in self.lift_centers
                               and np.linalg.norm(self.lift_centers[sid]-xy)<160]
                    if near_lift: self.board=(near_lift[0],self.calls+40)
                    # Another switch for the same lift would just send it back: share the cooldown.
                    for other in self.triggers:
                        if other is not t and other['targets']==t['targets'] and not other['once']:
                            other['until']=max(other['until'],self.calls+300)
            elif id(t)!=self.trigger_goal or self.mode!='trigger': continue
            elif d>96: self.trigger_since=self.calls  # still travelling; the clock starts on arrival
            elif self.calls-self.trigger_since>150:
                # Unreachable in practice (e.g. inside an obstacle): set it aside for a while.
                t['until']=self.calls+800; self.trigger_goal=None; self.path=[]
        self.last_xy=xy.copy()
        if 0<=node[1]<self.visits.shape[0] and 0<=node[0]<self.visits.shape[1]:
            self.visits[node[1],node[0]]+=1
        self.recent.append((xy.copy(),last_action))
        for color,pos in self.keys:
            # Pickup radius: player (16) plus key (20) bounding boxes.
            if color not in self.held and np.max(np.abs(pos-xy))<36:
                self.held.add(color); self._update_locks(); self.path=[]
        collision=False
        if len(self.recent)==12 and not enemy_visible and yaw is not None:
            # Any movement counts: the teacher's stuck recovery mixes backing up and strafing.
            moving=sum(a in (0,1,4,5,7,13,14,15) for _,a in self.recent)
            if moving>=7 and np.linalg.norm(xy-self.recent[0][0])<10:
                ahead=xy+32*np.array([np.cos(np.radians(yaw)),np.sin(np.radians(yaw))])
                # Temporarily reject a failed edge and seek an alternate approach.
                self.temporary[self.grid(ahead)]=self.calls+96
                self.recent.clear(); self.path=[]; collision=True; self.collision_events+=1
        if not self.path or self.calls%64==1 or np.linalg.norm(self.world(self.path[0])-xy)>128 or collision:
            self.path=self.route(node)
        while len(self.path)>1 and np.linalg.norm(self.world(self.path[0])-xy)<20:
            self.path.pop(0)
        target=self.world(self.path[0]) if self.path else xy.copy()
        # Bounded line-of-sight lookahead avoids grid zigzags and corner cutting.
        for waypoint in self.path[:8]:
            point=self.world(waypoint)
            if np.linalg.norm(point-xy)>96 or not self.clear(xy,point): break
            target=point
        if self.goal is not None and np.linalg.norm(xy-self.goal)<72:
            target=self.goal.copy()
            switch=next((t for t in self.triggers if id(t)==self.trigger_goal and t['kind']=='switch'),None)
            if self.version=='improved2' and self.mode=='trigger' and switch is not None and (xy-switch['p'])@switch['normal']>0:
                # Close to a switch the bearing to its midpoint swings wildly; aim through
                # the wall instead so the player squares up and the USE ray hits it.
                target=switch['mid']-switch['normal']*48
        # Pressing USE on a moving door reverses it; hide use targets until it opens.
        if last_action in (12,13): self.use_wait=self.calls+12
        near=1e6
        for p,q,mid in self.use_lines if self.calls>=self.use_wait else ():
            d=np.linalg.norm(mid-xy)
            # Only use interactions ahead of the route, rather than arbitrary specials.
            if d<80 and np.dot(mid-xy,target-xy)>0:
                delta=q-p; t=np.clip(np.dot(xy-p,delta)/max(np.dot(delta,delta),1),0,1)
                near=min(near,np.linalg.norm(xy-(p+t*delta)))
        return target,near
