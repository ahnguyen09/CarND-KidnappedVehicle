/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "map.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	//set number of particles
	num_particles = 20;

	//initialize weights to the same size as number of particles
	weights.resize(num_particles,1.0);

	//calculate multivariable probability variables
	gauss_norm = (1/(2*M_PI*std[0]*std[1]));
	sig_xx = 2*std[0]*std[0];
	sig_yy = 2*std[1]*std[1];

	//create engine to generate random value
	default_random_engine gen;

	//Create normal(gaussian) distributions for x, y and theta based gps 
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i=0;i<num_particles;i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//create engine to generate random value
	default_random_engine gen;
	
	//Create gaussian nosie distributions for x, y and theta based on 0 mean
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	double delta_d = velocity*delta_t;
	double delta_theta = yaw_rate*delta_t;
	
	double vel_yawrate_ratio = 0;
	if (fabs(yaw_rate) >= 0.01) {vel_yawrate_ratio = velocity/yaw_rate;} 
	
	for (int i=0;i<particles.size();i++) {
		double x_i = particles[i].x;
		double y_i = particles[i].y;
		double theta_i = particles[i].theta;

		if (fabs(yaw_rate) < 0.01) {
			//update x,y based on velocity and heading (theta)
			particles[i].x = x_i + delta_d*cos(theta_i) + dist_x(gen);
			particles[i].y = y_i + delta_d*sin(theta_i) + dist_y(gen); 		
		} else {
			//update x,y,theta based on velocity and yaw_rate (theta dot)
			particles[i].x = x_i + vel_yawrate_ratio*(sin(theta_i+delta_theta)-sin(theta_i)) + dist_x(gen);
			particles[i].y = y_i + vel_yawrate_ratio*(cos(theta_i)-cos(theta_i+delta_theta)) + dist_y(gen);
			particles[i].theta = theta_i + delta_theta + dist_theta(gen);  	 	
		}
	}
}

void ParticleFilter::dataAssociation(const Map &map_landmarks, vector<LandmarkObs>& observations) {
	// TODO: Find the Landmark that is closest to each observed measurement and assign the landmark id to observation id.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto &observation : observations) {
		double min_distance = 9.9E9;
		double ox = observation.x;
		double oy = observation.y;
		for (auto &landmark : map_landmarks.landmark_list) {
			double lx = landmark.x_f;
			double ly = landmark.y_f;

			double distance = dist(ox,oy,lx,ly);
			if (distance<min_distance) {
				observation.id = landmark.id_i;
				min_distance = distance;
			}
		}
	}
}

void ParticleFilter::FilterLandmarks(vector<Particle> &particles,float filter_window[],const Map &map_landmarks,
	Map &filter_map_landmarks) {
	// helper function for update weights. Filter landmark based on position of particle position and sensor distance
	
	// find min and max position of all particles
	float min_x = 9.9E9;
	float max_x = -9.9E9;
	float min_y = 9.9E9;
	float max_y = -9.9E9;

	//for each particle
	for (auto &particle : particles) {
		if (particle.x < min_x) {min_x = particle.x;}
		if (particle.x > max_x) {max_x = particle.x;}
		if (particle.y < min_y) {min_y = particle.y;}
		if (particle.y > max_y) {max_y = particle.y;}
	}

	float min_window_x = min_x - filter_window[0];
	float max_window_x = max_x + filter_window[0];
	float min_window_y = min_y - filter_window[1];
	float max_window_y = max_y + filter_window[1];

	//for each landmark
	for (auto &landmark : map_landmarks.landmark_list) {
		if (landmark.x_f >= min_window_x && landmark.x_f <= max_window_x &&
		    landmark.y_f >= min_window_y && landmark.y_f <= max_window_y) {
			//Map::single_landmark_s tempLandmark = landmark;
			filter_map_landmarks.landmark_list.push_back(landmark);
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], double std_pos[], 
		const vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Transforme observations from each particle to real world map perspective
	// update weights of each particle based on its observation compared to car observation
	
	// landmark position -> mu_x,mu_y
	// observation position (particle pos + obs distance transformed) -> x,y
	// transformed obs id hold closes landmark id
	
	float dx_filter = sensor_range + 4*std_pos[0];
	float dy_filter = sensor_range + 4*std_pos[1];
	Map filtered_map;

	// we should only care about landmark within sensor range (plus noise) of gps particle cluster
	// e.g. don't care about landmarks in China while driving in CA
	float filter_window[] = {dx_filter,dy_filter};
	FilterLandmarks(particles,filter_window,map_landmarks,filtered_map);

	// for each particle
	for (int i =0;i<particles.size();i++) {
		vector<LandmarkObs> transformed_observations;
		// for each observation
		for (auto &observation : observations) {
			//calculate transformation of particle to real word map coordinate
			LandmarkObs tempObs = observation;
			double x_obs = observation.x;
			double y_obs = observation.y;
			double theta = particles[i].theta;

			//tranform
			tempObs.x = particles[i].x + cos(theta)*x_obs - sin(theta)*y_obs;
			tempObs.y = particles[i].y + sin(theta)*x_obs + cos(theta)*y_obs;
			transformed_observations.push_back(tempObs); 
		}
		dataAssociation(filtered_map,transformed_observations);

		particles[i].weight = 1.0; //reset weight before applying products
		for (auto &trans_obs : transformed_observations) {
			// landmark position -> mu_x,mu_y
			// observation position (particle pos + obs distance transformed) -> x,y
			// landmark maping: id_i - 1 is index in list
			double mu_x = map_landmarks.landmark_list[int(trans_obs.id - 1)].x_f;
			double mu_y = map_landmarks.landmark_list[int(trans_obs.id - 1)].y_f;
			double x_obs = trans_obs.x;
			double y_obs = trans_obs.y;

			/* defined in init step
			gauss_norm = (1/(2*M_PI*std[0]*std[1]));
			sig_xx = 2*std[0]*std[0];
			sig_yy = 2*std[1]*std[1]; */

			double exponent = pow((x_obs - mu_x),2)/sig_xx + pow((y_obs - mu_y),2)/sig_yy;

			particles[i].weight *= gauss_norm * exp(-exponent);
		}

		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;
	random_device rd;
	mt19937 gen(rd());

	//converts weights vector to a list of weights that can be used for discrete distribution: (weights.begin(),weights.end())
	discrete_distribution<> d(weights.begin(),weights.end());
	for (int i=0;i<particles.size();i++) {
		int idx = d(gen);
		new_particles.push_back(particles[idx]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, vector<int> associations, vector<double> sense_x, vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
