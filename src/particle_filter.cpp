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
#include <limits>
#include <iterator>
#include <random>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  // Initialize normal distributions for x, y and theta.
  std::normal_distribution<double> x_normal{x, std[0]};
  std::normal_distribution<double> y_normal{y, std[1]};
  std::normal_distribution<double> theta_normal{theta, std[2]};
  
  // Set number of particles.
  num_particles = 100;
  
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = x_normal(gen);
    particle.y = y_normal(gen);
    particle.theta = theta_normal(gen);
    particle.weight = 1;

    particles.push_back(particle);
    weights.push_back(1);
  }
  
  // Set is_initialized to true.
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
  double measurement_x, measurement_y;
  double measurement_theta = 0;
  // Let's use the formula from Udacity's class to compute measurement/predictions.
  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) > 0.00001) {
      measurement_x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      measurement_y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      measurement_theta = particles[i].theta + yaw_rate*delta_t;
    }
    else {
      measurement_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
      measurement_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
      measurement_theta = particles[i].theta;
    }
    
    // Initialize normal distributions for x, y and theta predictions.
    std::normal_distribution<double> x_normal(measurement_x, std_pos[0]);
    std::normal_distribution<double> y_normal(measurement_y, std_pos[1]);
    std::normal_distribution<double> theta_normal(measurement_theta, std_pos[2]);
    
    // Final predictions
    particles[i].x = x_normal(gen);
    particles[i].y = y_normal(gen);
    particles[i].theta = theta_normal(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  
  // Run simple nearest neighbour algorithm.
  for (int i = 0; i < observations.size(); ++i) {
    // min_distance to keep track of minimum distance between a particular landmark and observation.
    double min_distance = std::numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); ++j) {
      double current_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (current_distance < min_distance) {
        min_distance = current_distance;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
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

  for (int i = 0; i < num_particles; ++i) {
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    std::vector<LandmarkObs> observations_map;

    // Converting landmark observations from car to map coordinates.
    for (int j = 0; j < observations.size(); ++j) {
      LandmarkObs landmark_obs;
      landmark_obs.x = particles[i].x + (cos(particles[i].theta)*observations[j].x) - (sin(particles[i].theta)*observations[j].y);
      landmark_obs.y = particles[i].y + (sin(particles[i].theta)*observations[j].x) + (cos(particles[i].theta)*observations[j].y);
      observations_map.push_back(landmark_obs);
    }

    particles[i].weight = 1.0;
    
    // Find nearest landmark and calculate weight.
    for (int j = 0; j < observations_map.size(); ++j){
      double min_distance = sensor_range;
      int association = -1;

      for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
        double distance = dist(observations_map[j].x, observations_map[j].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
        if (distance < min_distance) {
          min_distance = distance;
          association = k;
        }
      }
      
      // Found a nearby landmark.
      if(association >= 0){
        double x_f = map_landmarks.landmark_list[association].x_f;
        double y_f = map_landmarks.landmark_list[association].y_f;
        // Calculate normalization term
        long double gauss_norm = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
        // Calculate exponent
        long double exponent = -0.5 * (pow((observations_map[j].x-x_f)/std_landmark[0], 2) + pow((observations_map[j].y-y_f)/std_landmark[1], 2));
        // calculate weight using normalization terms and exponent
        long double multiplier = gauss_norm * exp(exponent);
        if (multiplier > 0) {
          particles[i].weight *= multiplier;
        }
      }
      
      // save the association and observations
      associations.push_back(association+1);
      sense_x.push_back(observations_map[j].x);
      sense_y.push_back(observations_map[j].y);
    }
    // Set new associations.
    particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
    // Set weights.
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  vector<Particle> resampled_particles;
  for(int i = 0; i < num_particles; ++i) {
    resampled_particles.push_back(particles[distribution(gen)]);
  }
  // save resampled particles
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
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
