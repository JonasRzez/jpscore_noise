/**
 * File:   ForceModel.cpp
 *
 * Created on 13. December 2010, 15:05
 *
 * @section LICENSE
 * This file is part of JuPedSim.
 *
 * JuPedSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * JuPedSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JuPedSim. If not, see <http://www.gnu.org/licenses/>.
 *
 * @section DESCRIPTION
 *
 *
 *
 */

#include "ForceModel.h"

/************************************************************
 ForceModel
 ************************************************************/

ForceModel::ForceModel() {
}


ForceModel::~ForceModel() {
}

/************************************************************
 GCFM ForceModel
 ************************************************************/

// Private Funktionen

/* treibende Kraft
 * Parameter:
 *   - ped: Fußgänger für den die Kraft berechnet wird
 *   - room: Raum (mit SubRooms) in dem das Ziel gesucht wird
 * Rückgabewerte:
 *   - Vektor(x,y) zum Ziel
 * */
inline Point GCFMModel::ForceDriv(Pedestrian* ped, Room* room) const {
	const Point& target = pdirection->GetTarget(room, ped);
	Point F_driv;
	const Point& pos = ped->GetPos();
	double dist = ped->GetExitLine()->DistTo(pos);

	if (dist > EPS_GOAL) {
		const Point& v0 = ped->GetV0(target);
		F_driv = ((v0 * ped->GetV0Norm() - ped->GetV()) * ped->GetMass()) / ped->GetTau();
	} else {
		const Point& v0 = ped->GetV0();
		F_driv = ((v0 * ped->GetV0Norm() - ped->GetV()) * ped->GetMass()) / ped->GetTau();
	}
	return F_driv;
}

/* abstoßende Kraft zwischen ped1 und ped2
 * Parameter:
 *   - ped1: Fußgänger für den die Kraft berechnet wird
 *   - ped2: Fußgänger mit dem die Kraft berechnet wird
 * Rückgabewerte:
 *   - Vektor(x,y) mit abstoßender Kraft
 * */
Point GCFMModel::ForceRepPed(Pedestrian* ped1, Pedestrian* ped2) const {

	Point F_rep;
	// x- and y-coordinate of the distance between p1 and p2
	Point distp12 = ped2->GetPos() - ped1->GetPos();
	const Point& vp1 = ped1->GetV(); // v Ped1
	const Point& vp2 = ped2->GetV(); // v Ped2
	Point ep12; // x- and y-coordinate of the normalized vector between p1 and p2
	double tmp, tmp2;
	double v_ij;
	double K_ij;
	//double r1, r2;
	double nom; //nominator of Frep
	double px; // hermite Interpolation value
	const Ellipse& E1 = ped1->GetEllipse();
	const Ellipse& E2 = ped2->GetEllipse();
	double distsq;
	double dist_eff = E1.EffectiveDistanceToEllipse(E2, &distsq);


	//          smax    dist_intpol_left      dist_intpol_right       dist_eff_max
	//       ----|-------------|--------------------------|--------------|----
	//       5   |     4       |            3             |      2       | 1

	// If the pedestrian is outside the cutoff distance, the force is zero.
	if (dist_eff >= pDistEffMaxPed) {
		F_rep = Point(0.0, 0.0);
		return F_rep;
	}
	//Point AP1inE1 = Point(E1.GetXp(), 0); // ActionPoint von E1 in Koordinaten von E1
	//Point AP2inE2 = Point(E2.GetXp(), 0); // ActionPoint von E2 in Koordinaten von E2
	// ActionPoint von E1 in Koordinaten von E2 (transformieren)
	//Point AP1inE2 = AP1inE1.CoordTransToEllipse(E2.GetCenter(), E2.GetCosPhi(), E2.GetSinPhi());
	// ActionPoint von E2 in Koordinaten von E1 (transformieren)
	//Point AP2inE1 = AP2inE2.CoordTransToEllipse(E1.GetCenter(), E1.GetCosPhi(), E1.GetSinPhi());
	//r1 = (AP1inE1 - E1.PointOnEllipse(AP2inE1)).Norm();
	//r2 = (AP2inE2 - E2.PointOnEllipse(AP1inE2)).Norm();

	//%------- Free parameter --------------
	Point p1, p2; // "Normale" Koordinaten
	double mindist;


	p1 = Point(E1.GetXp(), 0).CoordTransToCart(E1.GetCenter(), E1.GetCosPhi(), E1.GetSinPhi());
	p2 = Point(E2.GetXp(), 0).CoordTransToCart(E2.GetCenter(), E2.GetCosPhi(), E2.GetSinPhi());
	distp12 = p2 - p1;
	mindist = E1.MinimumDistanceToEllipse(E2); //ONE
	double dist_intpol_left = mindist + pintp_widthPed; // lower cut-off for Frep (modCFM)
	double dist_intpol_right = pDistEffMaxPed - pintp_widthPed; //upper cut-off for Frep (modCFM)
	double smax = mindist - pintp_widthPed; //max overlappig
	double f = 0.0f, f1 = 0.0f; //function value and its derivative at the interpolation point'

	//todo: runtime normsquare?
	if (distp12.Norm() >= EPS) {
		ep12 = distp12.Normalized();
	} else {
		Log->write("ERROR: \tin GCFMModel::forcePedPed() ep12 kann nicht berechnet werden!!!\n");
		Log->write("ERROR:\t fix this as soon as possible");
		return F_rep; // FIXME: should never happen
		exit(0);

	}
	// calculate the parameter (whatever dist is)
	tmp = (vp1 - vp2).ScalarP(ep12); // < v_ij , e_ij >
	v_ij = 0.5 * (tmp + fabs(tmp));
	tmp2 = vp1.ScalarP(ep12); // < v_i , e_ij >

	//todo: runtime normsquare?
	if (vp1.Norm() < EPS) { // if(norm(v_i)==0)
		K_ij = 0;
	} else {
		double bla = tmp2 + fabs(tmp2);
		K_ij = 0.25 * bla * bla / vp1.ScalarP(vp1); //squared

		if (K_ij < EPS * EPS) {
			F_rep = Point(0.0, 0.0);
			return F_rep;
		}
	}
	nom = pNuPed * ped1->GetV0Norm() + v_ij; // Nu: 0=CFM, 0.28=modifCFM;
	nom *= nom;

	K_ij = sqrt(K_ij);
	if (dist_eff <= smax) { //5
		f = -ped1->GetMass() * K_ij * nom / dist_intpol_left;
		F_rep = ep12 * pmaxfPed * f;
		return F_rep;
	}

	//          smax    dist_intpol_left      dist_intpol_right       dist_eff_max
	//	     ----|-------------|--------------------------|--------------|----
	//       5   |     4       |            3             |      2       | 1

	if (dist_eff >= dist_intpol_right) { //2
		f = -ped1->GetMass() * K_ij * nom / dist_intpol_right; // abs(NR-Dv(i)+Sa)
		f1 = -f / dist_intpol_right;
		px = hermite_interp(dist_eff, dist_intpol_right, pDistEffMaxPed, f, 0, f1, 0);
		F_rep = ep12 * px;
	} else if (dist_eff >= dist_intpol_left) { //3
		f = -ped1->GetMass() * K_ij * nom / fabs(dist_eff); // abs(NR-Dv(i)+Sa)
		F_rep = ep12 * f;
	} else {//4
		f = -ped1->GetMass() * K_ij * nom / dist_intpol_left;
		f1 = -f / dist_intpol_left;
		px = hermite_interp(dist_eff, smax, dist_intpol_left, pmaxfPed*f, f, 0, f1);
		F_rep = ep12 * px;
	}
	if (F_rep.GetX() != F_rep.GetX() || F_rep.GetY() != F_rep.GetY()) {
		char tmp[CLENGTH];
		sprintf(tmp, "\nNAN return ----> p1=%d p2=%d Frepx=%f, Frepy=%f\n", ped1->GetPedIndex(),
				ped2->GetPedIndex(), F_rep.GetX(), F_rep.GetY());
		Log->write(tmp);
		Log->write("ERROR:\t fix this as soon as possible");
		return Point(0,0); // FIXME: should never happen
		exit(0);
	}
	return F_rep;
}

/* abstoßende Kraft zwischen ped und subroom
 * Parameter:
 *   - ped: Fußgänger für den die Kraft berechnet wird
 *   - subroom: SubRoom für den alle abstoßende Kräfte von Wänden berechnet werden
 * Rückgabewerte:
 *   - Vektor(x,y) mit Summe aller abstoßenden Kräfte im SubRoom
 * */

inline Point GCFMModel::ForceRepRoom(Pedestrian* ped, SubRoom* subroom) const {
	Point f = Point(0., 0.);
	//first the walls
	const vector<Wall>& walls = subroom->GetAllWalls();
	for (int i = 0; i < subroom->GetAnzWalls(); i++) {
		f = f + ForceRepWall(ped, walls[i]);
	}

	//then the obstacles
	const vector<Obstacle*>& obstacles = subroom->GetAllObstacles();
	for(unsigned int obs=0;obs<obstacles.size();++obs){
		const vector<Wall>& walls = obstacles[obs]->GetAllWalls();
		for (unsigned int i = 0; i < walls.size(); i++) {
			f = f + ForceRepWall(ped, walls[i]);
		}
	}

	//eventually crossings
	const vector<Crossing*>& crossings = subroom->GetAllCrossings();
	for (unsigned int i = 0; i < crossings.size(); i++) {
		Crossing* goal=crossings[i];
		int uid1= goal->GetUniqueID();
		int uid2=ped->GetExitLine()->GetUniqueID();
		// ignore my transition
		if (uid1 != uid2) {
			f = f + ForceRepWall(ped,*((Wall*)goal));
		}
	}

	// and finally the closed doors
	const vector<Transition*>& transitions = subroom->GetAllTransitions();
	for (unsigned int i = 0; i < transitions.size(); i++) {
		Transition* goal=transitions[i];
		int uid1= goal->GetUniqueID();
		int uid2=ped->GetExitLine()->GetUniqueID();
		// ignore my transition consider closed doors
		//closed doors are considered as wall
		if( (goal->IsOpen()==false) || (uid1 != uid2) ) {
			f = f + ForceRepWall(ped,*((Wall*)goal));
		}
	}

	return f;
}

/* abstoßende Kraft zwischen ped und Wand, dazu werden drei Punktkräfte benutzt
 * Parameter:
 *   - ped: Fußgänger für den die Kraft berechnet wird
 *   - w: Wand mit der abstoßende Kraft berechnet wird
 * Rückgabewerte:
 *   - Vektor(x,y) mit abstoßender Kraft zur Wand
 * */

inline Point GCFMModel::ForceRepWall(Pedestrian* ped, const Wall& w) const {
	Point F = Point(0.0, 0.0);
	Point pt = w.ShortestPoint(ped->GetPos());
	double wlen = w.LengthSquare();
	if (wlen < 0.01) { // smaller than 10 cm
		return F; //ignore small lines
	}
	// Kraft soll nur orthgonal wirken
	if (fabs((w.GetPoint1() - w.GetPoint2()).ScalarP(ped->GetPos() - pt)) > EPS)
		return F;

	double mind = ped->GetEllipse().MinimumDistanceToLine(w);
	double vn = w.NormalComp(ped->GetV()); //normal component of the velocity on the wall
	return  ForceRepStatPoint(ped, pt, mind, vn); //line --> l != 0
}

/* abstoßende Punktkraft zwischen ped und Punkt p
 * Parameter:
 *   - ped: Fußgänger für den die Kraft berechnet wird
 *   - p: Punkt von dem die Kaft wirkt
 *   - l: Parameter zur Käfteinterpolation
 *   - vn: Parameter zur Käfteinterpolation
 * Rückgabewerte:
 *   - Vektor(x,y) mit abstoßender Kraft
 * */

Point GCFMModel::ForceRepStatPoint(Pedestrian* ped, const Point& p, double l, double vn) const {
	Point F_rep = Point(0.0, 0.0);
	const Point& v = ped->GetV();
	Point dist = p - ped->GetPos(); // x- and y-coordinate of the distance between ped and p
	double d = dist.Norm(); // distance between the centre of ped and point p
	Point e_ij; // x- and y-coordinate of the normalized vector between ped and p
	double K_ij;
	double tmp;
	double bla;
	Point r;
	Point pinE; // vorher x1, y1
	const Ellipse& E = ped->GetEllipse();

	if (d < EPS)
		return Point(0.0, 0.0);
	e_ij = dist / d;
	tmp = v.ScalarP(e_ij); // < v_i , e_ij >;
	bla = (tmp + fabs(tmp));
	if (!bla) // Fussgaenger nicht im Sichtfeld
		return Point(0.0, 0.0);
	if (fabs(v.GetX()) < EPS && fabs(v.GetY()) < EPS) // v==0)
		return Point(0.0, 0.0);
	K_ij = 0.5 * bla / v.Norm(); // K_ij
	// Punkt auf der Ellipse
	pinE = p.CoordTransToEllipse(E.GetCenter(), E.GetCosPhi(), E.GetSinPhi());
	// Punkt auf der Ellipse
	r = E.PointOnEllipse(pinE);
	//interpolierte Kraft
	F_rep = ForceInterpolation(ped->GetV0Norm(), K_ij, e_ij, vn, d, (r - E.GetCenter()).Norm(), l);
	return F_rep;
}

Point GCFMModel::ForceInterpolation(double v0, double K_ij, const Point& e, double vn, double d, double r, double l) const {
	Point F_rep;
	double nominator = pNuWall * v0 + vn;
	nominator *= nominator*K_ij;
	double f = 0, f1 = 0; //function value and its derivative at the interpolation point
	//BEGIN ------- interpolation parameter
	double smax = l - pintp_widthWall; // max overlapping radius
	double dist_intpol_left = l + pintp_widthWall; //r_eps
	double dist_intpol_right = pDistEffMaxWall - pintp_widthWall;
	//END ------- interpolation parameter

	double dist_eff = d - r;

	//         smax    dist_intpol_left      dist_intpol_right       dist_eff_max
	//	     ----|-------------|--------------------------|--------------|----
	//       5   |     4       |            3             |      2       | 1

	double px = 0; //value of the interpolated function
	double tmp1 = pDistEffMaxWall;
	double tmp2 = dist_intpol_right;
	double tmp3 = dist_intpol_left;
	double tmp5 = smax + r;

	if (dist_eff >= tmp1) { // 1
		//F_rep = Point(0.0, 0.0);
		return F_rep;
	}

	if (dist_eff <= tmp5) { // 5
		F_rep = e * (-pmaxfWall);
		return F_rep;
	}

	if (dist_eff > tmp2) { //2
		f = -nominator / dist_intpol_right;
		f1 = -f / dist_intpol_right; // nominator / (dist_intpol_right^2) = derivativ of f
		px = hermite_interp(dist_eff, dist_intpol_right, pDistEffMaxWall, f, 0, f1, 0);
		F_rep = e * px;
	} else if (dist_eff >= tmp3) { //3
		f = -nominator / fabs(dist_eff); //devided by abs f the effective distance
		F_rep = e * f;
	} else { //4 d > smax FIXME
		f = -nominator / dist_intpol_left;
		f1 = -f / dist_intpol_left;
		px = hermite_interp(dist_eff, smax, dist_intpol_left, pmaxfWall*f, f, 0, f1);
		F_rep = e * px;
	}
	return F_rep;
}

// Konstruktoren

GCFMModel::GCFMModel(DirectionStrategy* dir, double nuped, double nuwall, double dist_effPed,
		double dist_effWall, double intp_widthped, double intp_widthwall, double maxfped,
		double maxfwall) {
	pdirection = dir;
	pNuPed = nuped;
	pNuWall = nuwall;
	pintp_widthPed = intp_widthped;
	pintp_widthWall = intp_widthwall;
	pmaxfPed = maxfped;
	pmaxfWall = maxfwall;
	pDistEffMaxPed = dist_effPed;
	pDistEffMaxWall = dist_effWall;

}

GCFMModel::~GCFMModel(void) {
	// pdirection wird in Simulation freigegeben
}

// Getter-Funktionen

DirectionStrategy* GCFMModel::GetDirection() const {
	return pdirection;
}

double GCFMModel::GetNuPed() const {
	return pNuPed;
}

double GCFMModel::GetNuWall() const {
	return pNuWall;
}

double GCFMModel::GetIntpWidthPed() const {
	return pintp_widthPed;
}

double GCFMModel::GetIntpWidthWall() const {
	return pintp_widthWall;
}

double GCFMModel::GetMaxFPed() const {
	return pmaxfPed;
}

double GCFMModel::GetMaxFWall() const {
	return pmaxfWall;
}

double GCFMModel::GetDistEffMaxPed() const {
	return pDistEffMaxPed;
}

double GCFMModel::GetDistEffMaxWall() const {
	return pDistEffMaxWall;
}

string GCFMModel::writeParameter() const {
	string rueck;
	char tmp[CLENGTH];

	sprintf(tmp, "\t\tNu: \t\tPed: %f \tWall: %f\n", pNuPed, pNuWall);
	rueck.append(tmp);
	sprintf(tmp, "\t\tInterp. Width: \tPed: %f \tWall: %f\n", pintp_widthPed, pintp_widthWall);
	rueck.append(tmp);
	sprintf(tmp, "\t\tMaxF: \t\tPed: %f \tWall: %f\n", pmaxfPed, pmaxfWall);
	rueck.append(tmp);
	sprintf(tmp, "\t\tDistEffMax: \tPed: %f \tWall: %f\n", pDistEffMaxPed, pDistEffMaxWall);
	rueck.append(tmp);

	return rueck;
}


// virtuelle Funktionen

/* berechnet die Kräfte nach dem GCFM Modell
 * Parameter:
 *   - time:
 *   - building:
 *   - roomID:
 *   -subroomID:
 * Rückgabewerte:
 *   - result_acc:
 * */
void GCFMModel::CalculateForce(double time, vector< Point >& result_acc, Building* building,
		int roomID, int subroomID) const {
	double delta = 0.5;
	Room* room = building->GetRoom(roomID);
	SubRoom* subroom = room->GetSubRoom(subroomID);
	vector<Pedestrian*> allpeds = subroom->GetAllPedestrians(); //All the peds in this room


	vector<Pedestrian*> allpeds_otherrooms = vector<Pedestrian*>(); //All the peds in the neighbouring rooms
	vector<Crossing*> rep_goals = vector<Crossing*>();
	vector<Point> fdriv = vector<Point > (); //Driven force.
	vector<Point> frep = vector<Point > (); //Repulsive pedestrian force
	vector<Point> fwall = vector<Point > (); //Repulsive from wall


	for (int p = 0; p < (int) allpeds.size(); ++p) {
		Pedestrian* ped = allpeds[p];
		double normVi = ped->GetV().ScalarP(ped->GetV());
		double tmp = (ped->GetV0Norm() + delta) * (ped->GetV0Norm() + delta);
		if (normVi > tmp && ped->GetV0Norm() > 0) {
			char tmp[CLENGTH];
			sprintf(tmp, "WARNING:\t GCFMModel::calculateForce() actual velocity (%f) of iped %d "
					"[%d/%d] is bigger than desired velocity (%f) at time: %f s\n",
					sqrt(normVi), ped->GetPedIndex(), ped->GetRoomID(), ped->GetSubRoomID(),
					ped->GetV0Norm(), time);
			Log->write(tmp);
			exit(0);
		}
	}


	// Driving Force
	for (int p = 0; p < (int) allpeds.size(); ++p) {
		Point temp = ForceDriv(allpeds[p], room);
		fdriv.push_back(temp);
	}

	//Abstoßende Kräfte zu anderen Fußgängern
	// Crossings und Transitions durchgehen
	for (int i = 0; i < building->GetRouting()->GetAnzGoals(); i++) {
		Crossing* goal = building->GetRouting()->GetGoal(i);
		if (goal->GetSubRoom1() != goal->GetSubRoom2()) {
			if (goal->IsInRoom(roomID) && goal->IsInSubRoom(subroomID)) {
				rep_goals.push_back(goal); // alle goals des Raums speichern
				SubRoom* other_subroom = goal->GetOtherSubRoom(roomID, subroomID);
				if (other_subroom != NULL) {
					vector<Pedestrian*> other_peds = other_subroom->GetAllPedestrians();
					allpeds_otherrooms.insert(allpeds_otherrooms.end(), other_peds.begin(), other_peds.end());
				}
			}
		}
	}
	for (int p1 = 0; p1 < (int) allpeds.size(); p1++) {
		Point F_rep;
		for (int p2 = 0; p2 < (int) allpeds.size(); p2++) {
			if (p1 != p2) {
				Point tmp;
				tmp = ForceRepPed(allpeds[p1], allpeds[p2]);
				F_rep = F_rep + tmp;
			}
		}

		//And also the force between peds in neighbouring rooms
		for (int p2 = 0; p2 < (int) allpeds_otherrooms.size(); p2++) {
			Point tmp = ForceRepPed(allpeds[p1], allpeds_otherrooms[p2]);
			F_rep = F_rep + tmp;
		}
		frep.push_back(F_rep);
	}

	// Abstoßende Kräfte zu Wänden
	for (int p = 0; p < (int) allpeds.size(); p++) {
		// "normale" Wände
		Point repwall = ForceRepRoom(allpeds[p], subroom);
		// alle Crossings/Transitons außer der eigenen wirken abstoßend
		for (int i = 0; i < (int) rep_goals.size(); i++) {
			if (rep_goals[i]->GetIndex() != allpeds[p]->GetExitIndex()) {
				Point tmp = ForceRepWall(allpeds[p], Wall(rep_goals[i]->GetPoint1(), rep_goals[i]->GetPoint2()));
				repwall = repwall + tmp;
			}
		}

		fwall.push_back(repwall);
	}

	// Addition
	for (int i = 0; i < (int) allpeds.size(); ++i) {
		result_acc.push_back((fdriv[i] + frep[i] + fwall[i]) / allpeds[i]->GetMass());
	}

	allpeds_otherrooms.clear();
	fdriv.clear();
	frep.clear();
	fwall.clear();
}

/**
 * implementation of Linked-cell combined with openMP
 */

void GCFMModel::CalculateForceLC(double time, double tip1, Building* building) const {
	double delta = 0.5;
	double h = tip1 - time;

	// collect all pedestrians in the simulation.
	const vector< Pedestrian* >& allPeds = building->GetAllPedestrians();

	unsigned int nSize = allPeds.size();

	int nThreads = 1;

#ifdef _OPENMP
	 nThreads = omp_get_max_threads();
#endif

	// check if worth sharing the work
	if (nSize < 20) nThreads = 1;
	int partSize = nSize / nThreads;

#pragma omp parallel  default(shared) num_threads(nThreads)
	{
		vector< Point > result_acc = vector<Point > ();
		result_acc.reserve(2200);

		int threadID = omp_get_thread_num();
		int start = threadID*partSize;
		int end = (threadID + 1) * partSize - 1;
		if ((threadID == nThreads - 1)) end = nSize - 1;

		for (int p = start; p <= end; ++p) {

			Pedestrian* ped = allPeds[p];
			Room* room = building->GetRoom(ped->GetRoomID());
			SubRoom* subroom = room->GetSubRoom(ped->GetSubRoomID());

			double normVi = ped->GetV().ScalarP(ped->GetV());
			double tmp = (ped->GetV0Norm() + delta) * (ped->GetV0Norm() + delta);
			if (normVi > tmp && ped->GetV0Norm() > 0) {
				fprintf(stderr, "GCFMModel::calculateForce() WARNING: actual velocity (%f) of iped %d "
						"is bigger than desired velocity (%f) at time: %fs\n",
						sqrt(normVi), ped->GetPedIndex(), ped->GetV0Norm(), time);


				//remove the pedestrian and continue
				for(int p=0;p<subroom->GetAnzPedestrians();p++){
					if (subroom->GetPedestrian(p)->GetPedIndex()==ped->GetPedIndex()){
						subroom->DeletePedestrian(p);
						break;
					}
				}

				building->DeletePedestrian(ped);
				Log->write("\tCRITICAL: one ped was removed due to high velocity");

				continue;
				//exit(0);
			}

			Point F_rep;
			//vector<Pedestrian*> neighbours;
			Pedestrian* neighbours[300]={NULL};
			int nSize=0;
			building->GetGrid()->GetNeighbourhood(ped,neighbours,&nSize);

			for (int i = 0; i < nSize; i++) {
				Pedestrian* ped1 = neighbours[i];
				//if they are in the same subroom
				if (ped->GetUniqueRoomID() == ped1->GetUniqueRoomID()) {
					F_rep = F_rep + ForceRepPed(ped, ped1);
				} else {
					// or in neighbour subrooms
					SubRoom* sb2=building->GetRoom(ped1->GetRoomID())->GetSubRoom(ped1->GetSubRoomID());
					if(subroom->IsDirectlyConnectedWith(sb2)){
						F_rep = F_rep + ForceRepPed(ped, ped1);
					}
				}
			}

			//repulsive forces to the walls and transitions that are not my target
			Point repwall = ForceRepRoom(allPeds[p], subroom);

			Point acc = (ForceDriv(ped, room) + F_rep + repwall) / ped->GetMass();
			result_acc.push_back(acc);
		}

		//#pragma omp barrier
		// update
		for (int p = start; p <= end; ++p) {
			Pedestrian* ped = allPeds[p];
			Point v_neu = ped->GetV() + result_acc[p - start] * h;
			Point pos_neu = ped->GetPos() + v_neu * h;

			//Jam is based on the current velocity
			if (v_neu.Norm() >= EPS_V){
				ped->ResetTimeInJam();
			}else{
				ped->UpdateTimeInJam();
			}

			ped->SetPos(pos_neu);
			ped->SetV(v_neu);
			ped->SetPhiPed();
		}

	}//end parallel
}
