#include "ros/ros.h"
#include "ros/time.h"
#include "sensor_msgs/Imu.h"
#include "trajectory_msgs/JointTrajectory.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/Float64.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/MultiArrayLayout.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/PointStamped.h"
#include "control_msgs/JointTrajectoryControllerState.h"
#include <cstdlib>
#include <tgmath.h> 
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include "control_msgs/FollowJointTrajectoryActionGoal.h"

#include <string>

#include <termios.h>            //termios, TCSANOW, ECHO, ICANON

#include <std_srvs/Empty.h>

#include <math.h>

#include <cmath>



// https://github.com/sdipendra/ros-projects/blob/master/src/keyboard_non_blocking_input/src/keyboard_non_blocking_input_node.cpp
char getch()
{
    fd_set set;
    struct timeval timeout;
    int rv;
    char buff = 0;
    int len = 1;
    int filedesc = 0;
    FD_ZERO(&set);
    FD_SET(filedesc, &set);

    timeout.tv_sec = 0;
    timeout.tv_usec = 1000;

    rv = select(filedesc + 1, &set, NULL, NULL, &timeout);

    struct termios old = {0};
    if (tcgetattr(filedesc, &old) < 0)
        ROS_ERROR("tcsetattr()");
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(filedesc, TCSANOW, &old) < 0)
        ROS_ERROR("tcsetattr ICANON");

    if(rv == -1)
        ROS_ERROR("select");
    else if(rv == 0)
        int x {0};
    else
        read(filedesc, &buff, len );

    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(filedesc, TCSADRAIN, &old) < 0)
        ROS_ERROR ("tcsetattr ~ICANON");
    return (buff);
}


//========================================================================================//
//===========================CLASS DECLARATION============================================//
//========================================================================================//
class ASDisturbanceController
{
    public:
        ASDisturbanceController(ros::NodeHandle &nh)
        {
            this->nh_ = nh;
            this->base_imu_sub_ = this->nh_.subscribe("/as/base_imu", 50, &ASDisturbanceController::base_imu_cb, this); // this is the fix 
            this->ee_imu_sub_ = this->nh_.subscribe("/as/ee_imu", 50, &ASDisturbanceController::ee_imu_cb, this);
            this->js_sub_ = this->nh_.subscribe("/as_control/xarm_position_trajectory_controller/state", 50, &ASDisturbanceController::joint_state_cb, this);
            this->curr_base_ori_.x = 0.0; 
            this->curr_base_ori_.y = 0.0; 
            this->curr_base_ori_.z = 0.0; 

            this->current_joint_state_.actual.positions.resize(6);
            this->current_joint_state_.actual.velocities.resize(6);
            this->current_joint_state_.actual.effort.resize(6);

            this->base_twist_pub_ = this->nh_.advertise<geometry_msgs::Twist>("/as_control/as_base_differential_controller/cmd_vel", 1000); 
            this->arm_pub_ = this->nh_.advertise<std_msgs::Float64MultiArray>("/as_control/xarm_joint_position_controller/command", 100);
            this->trans_error_pub_ = this->nh_.advertise<geometry_msgs::PointStamped>("/as_control/ee_translation_error", 1000);
            this->rot_error_pub_ = this->nh_.advertise<geometry_msgs::PointStamped>("/as_control/ee_orientation_error", 1000);
            std::cout << "Disturbance controller created!\n";
        }
        ~ASDisturbanceController() {std::cout << "Disturbance controller class destroyed.\n";}
        void set_desired_joint_values(std::vector<double> des_j_v);
        void set_desired_base_vel(std::vector<double> &b_v);
        void timestep();

    private:
        int num_joints_ {6};
        double rate_ {1 / 50};
        double alpha_ {1};
        double epsilon_ {0.01};
        int ctr_ {0};
        bool check_joint_vals();
        bool ready2control_ {false};
        std::vector<double> d_thet_ {0.0, 0.0, 0.0, 0.0, 0.0}; 
        int started_ {0};
        Eigen::Matrix4d base_pose_;
        std::vector<double> ee_velocities_from_IMU_ {0.0, 0.0, 0.0};
        robot_model_loader::RobotModelLoader robot_model_loader_{"robot_description"};
        robot_model::RobotModelPtr kinematic_model_ = robot_model_loader_.getModel();
        robot_state::RobotStatePtr kinematic_state_{new robot_state::RobotState(kinematic_model_)};
        const robot_state::JointModelGroup* joint_model_group_ = kinematic_model_->getJointModelGroup("as_xarm_grp");
        void control_protocol();
        Eigen::Matrix4d get_ee_pose_no_bumps();
        Eigen::Matrix4d get_ee_pose_wrt_base_footprint_frame();
        void update_base_pose_();
        void publish_the_errors(Eigen::VectorXd &err_six);
        Eigen::VectorXd homog_to_trans_and_euler(Eigen::Matrix4d &mat);
        Eigen::VectorXd get_error_vector(Eigen::VectorXd &desired, Eigen::VectorXd &actual);
        Eigen::MatrixXd get_jacobian();
        Eigen::VectorXd get_delta_theta(Eigen::MatrixXd &this_jacobian, Eigen::VectorXd &current_error);
        Eigen::Matrix4d get_actual_ee_pose();
        void control_timestep();
        void correct_joint_positions(Eigen::VectorXd &dee_theta);
        float x_position_ {0.0};
        float x_velocity_ {0.0};
        float x_acceleration_ {0.0};
        float y_position_ {0.0};
        float y_velocity_ {0.0};
        float y_acceleration_ {0.0};
        float z_position_ {0.0};
        float z_velocity_ {0.0};
        float z_acceleration_ {0.0};
        void ee_imu_cb(const sensor_msgs::Imu::ConstPtr &ee_imu_measurement);
        void joint_state_cb(const control_msgs::JointTrajectoryControllerState::ConstPtr &current_js);
        void base_imu_cb(const sensor_msgs::Imu::ConstPtr &base_imu_measurement);
        ros::Publisher base_twist_pub_;
        ros::Publisher arm_pub_;

        ros::Publisher trans_error_pub_;
        ros::Publisher rot_error_pub_;

        void set_base_vels();
        void set_arm();

        ros::NodeHandle nh_;
        ros::Subscriber base_imu_sub_;
        ros::Subscriber ee_imu_sub_;
        ros::Subscriber js_sub_;
        std::vector<double> desired_joint_positions_;
        std::vector<double> corrected_joint_positions_;

        std::vector<double> ee_angular_velocities_ {0.0, 0.0, 0.0, 0.0};

        std::vector<double> des_base_vel_;
        control_msgs::JointTrajectoryControllerState current_joint_state_;
        geometry_msgs::Pose desired_pose_wrt_base_frame_;
        geometry_msgs::Pose desired_pose_wrt_flat_base_frame_;

        std::vector<sensor_msgs::Imu> base_IMU_measurements_;
        sensor_msgs::Imu current_base_IMU_measurement_;
        geometry_msgs::Vector3 curr_base_ori_;
        std::vector<sensor_msgs::Imu> ee_IMU_measurements_;

        double custom_mod(double a, double n){
            return a - ( std::floor( ( a / n ) ) * n );
        }
};


//========================================================================================//
//===========================MAIN LOOP====================================================//
//========================================================================================//
int main(int argc, char **argv)
{
    ros::init(argc, argv, "earl_testing_node");

    std::cout << "====================================================\nCommencing testing node...\n====================================================\n";

    ros::NodeHandle nh;

    ASDisturbanceController my_controller = ASDisturbanceController(nh);

    std::vector<double> des_base_velocities {0.05, 0.0, 0.0};
    std::vector<double> des_j_pos {-1.4, 0.75, -1.4, 0.0, 0.58, 0.0};

    my_controller.set_desired_base_vel(des_base_velocities);
    my_controller.set_desired_joint_values(des_j_pos);

    ros::Rate rate(10);

    ros::Duration(4).sleep();

    int ctr {0};
    double r {1 / 50};

    ros::ServiceClient pauseGazebo = nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");
    std_srvs::Empty emptySrv;
    pauseGazebo.call(emptySrv);

    ros::Duration(4).sleep();

    std::cout << "====================================================\nPublishing velocity...\n====================================================\n";
    std::cout << "Press n to stop and y to start again (base movement)\n";


    // bool time2exit {false};
    while (ros::ok())
    { 
        ros::Duration d;

        double start_time {ros::Time::now().toSec()};
        my_controller.timestep();

        int c = 0;
        c = getch();

        if (c == 'n')
        {
            std::vector<double> replacement_base_vels {0.0, 0.0, 0.0};
            my_controller.set_desired_base_vel(replacement_base_vels);
            std::cout << "\n";
        } else if (c == 'y')
        {
            my_controller.set_desired_base_vel(des_base_velocities);
            std::cout << "\n";
        }
        


        ctr++;

        double time_diff {ros::Time::now().toSec() - start_time};
        ros::Duration((r - time_diff) * -1).sleep();
        ros::spinOnce();
    }
    return 0;
}



//========================================================================================//
//===========================FUNCTION DEFINITIONS=========================================//
//========================================================================================//
void ASDisturbanceController::base_imu_cb(const sensor_msgs::Imu::ConstPtr &base_imu_measurement)
{
    this->base_IMU_measurements_.push_back(*(base_imu_measurement));
    this->current_base_IMU_measurement_ = *(base_imu_measurement);
    
    if ( ( std::isnan( ((*base_imu_measurement)).linear_acceleration.x ) == false )
        &&
        ( std::isnan( ((*base_imu_measurement)).linear_acceleration.y ) == false )
        &&
        ( std::isnan( ((*base_imu_measurement)).linear_acceleration.z ) == false ) )
    {
        this->x_acceleration_ = (*(base_imu_measurement)).linear_acceleration.x;
        this->x_velocity_ = this->x_velocity_ + this->x_acceleration_;
        this->x_position_ = this->x_position_ + (this->x_velocity_);

        this->y_acceleration_ = (*(base_imu_measurement)).linear_acceleration.y;
        this->y_velocity_ = this->y_velocity_ + this->y_acceleration_;
        this->y_position_ = this->y_position_ + (this->y_velocity_);

        this->z_acceleration_ = (*(base_imu_measurement)).linear_acceleration.z;
        this->z_velocity_ = this->z_velocity_ + this->z_acceleration_;
        this->z_position_ = this->z_position_ + (this->z_velocity_);
    }
}

void ASDisturbanceController::ee_imu_cb(const sensor_msgs::Imu::ConstPtr &ee_imu_measurement)
{
    this->ee_IMU_measurements_.push_back(*(ee_imu_measurement));
    this->ee_velocities_from_IMU_.at(0) = this->ee_velocities_from_IMU_.at(0) + static_cast<double>( (*(ee_imu_measurement)).linear_acceleration.x );
    this->ee_velocities_from_IMU_.at(1) = this->ee_velocities_from_IMU_.at(0) + static_cast<double>( (*(ee_imu_measurement)).linear_acceleration.y );
    this->ee_velocities_from_IMU_.at(2) = this->ee_velocities_from_IMU_.at(0) + static_cast<double>( (*(ee_imu_measurement)).linear_acceleration.z );

    this->ee_angular_velocities_.at(0) = (*(ee_imu_measurement)).orientation.x;
    this->ee_angular_velocities_.at(1) = (*(ee_imu_measurement)).orientation.y;
    this->ee_angular_velocities_.at(2) = (*(ee_imu_measurement)).orientation.z;
    this->ee_angular_velocities_.at(3) = (*(ee_imu_measurement)).orientation.w;
}

void ASDisturbanceController::joint_state_cb(const control_msgs::JointTrajectoryControllerState::ConstPtr &current_js)
{
    this->current_joint_state_ = *(current_js);
}

void ASDisturbanceController::set_desired_base_vel(std::vector<double> &b_v)
{
    if (b_v.size() == 3)
    {
        this->des_base_vel_ = b_v;
    } else 
    {
        std::cout << "Input vector for desired linear base velocities is of inappropriate dimension.\n";
    }
}

void ASDisturbanceController::set_base_vels()
{
    geometry_msgs::Twist vel;
    vel.linear.x = this->des_base_vel_.at(0);
    vel.linear.y = this->des_base_vel_.at(1);
    vel.linear.z = this->des_base_vel_.at(2);
    this->base_twist_pub_.publish(vel);
}


void ASDisturbanceController::set_arm()
{
    std::vector<double> j_pos_to_pub;

    if (this->ready2control_ == false)
    {
        j_pos_to_pub = this->desired_joint_positions_;
    } else
    {
        j_pos_to_pub = this->corrected_joint_positions_;
    }

    std_msgs::Float64MultiArray arr;
    arr.data.clear();

    for (size_t i{0}; i < j_pos_to_pub.size(); i++)
    {
        arr.data.push_back(j_pos_to_pub.at(i));
    }

    this->arm_pub_.publish(arr);
}


void ASDisturbanceController::set_desired_joint_values(std::vector<double> des_j_v)
{
    if (des_j_v.size() == 6)
    {
        this->desired_joint_positions_ = des_j_v;
        this->corrected_joint_positions_ = this->desired_joint_positions_;
    } else
    {
        std::cout << "Input for desired joint values is of innappropriate size.\n";
    }
}


void ASDisturbanceController::timestep()
{

    if (this->ready2control_ == true && (this->ctr_ > 400))
    {
        this->control_protocol();
    }
    this->set_arm();
    if ((this->check_joint_vals() == true) && (this->ctr_ > 1500))
    {
        this->started_++;
        this->set_base_vels();
    }
    this->ctr_++;
}


bool ASDisturbanceController::check_joint_vals()
{
    for (int i {0}; i < 6; i++)
    {
        if ((this->current_joint_state_.actual.velocities[i]*this->current_joint_state_.actual.velocities[i]) > 0.001)
        {
            return false;
        }
    }
    this->ready2control_ = true;
    return true;
}


//========================================================================================//
//===========================ACTUAL CONTROLLER CODE=======================================//
//========================================================================================//

void ASDisturbanceController::control_protocol()
{
    if (this->started_ == 1)
    {
        std::cout << "Disturbance rejection control protocol initiated...\n";
    }
    this->update_base_pose_();
    this->control_timestep();
}

Eigen::Matrix4d ASDisturbanceController::get_ee_pose_no_bumps() 
{
    this->kinematic_state_->setJointGroupPositions(this->joint_model_group_, this->desired_joint_positions_);
    Eigen::Affine3d pos {this->kinematic_state_->getGlobalLinkTransform("xarmlink_eef")};
    Eigen::Matrix3d ppp {pos.linear()};
    Eigen::Matrix4d mat_2_ret;
    mat_2_ret <<    ppp(0,0), ppp(0,1), ppp(0,2), pos.translation().x(), 
                    ppp(1,0), ppp(1,1), ppp(1,2), pos.translation().y(), 
                    ppp(2,0), ppp(2,1), ppp(2,2), pos.translation().z(), 
                    0.0, 0.0, 0.0, 1.0;
    return mat_2_ret;
}

Eigen::Matrix4d ASDisturbanceController::get_actual_ee_pose()
{
    std::vector<double> curr_j_pos 
        {this->current_joint_state_.actual.positions[4],
        this->current_joint_state_.actual.positions[5],
        this->current_joint_state_.actual.positions[6],
        this->current_joint_state_.actual.positions[7],
        this->current_joint_state_.actual.positions[8],
        this->current_joint_state_.actual.positions[9]};

    this->kinematic_state_->setJointGroupPositions(this->joint_model_group_, this->corrected_joint_positions_);
    Eigen::Affine3d pos {this->kinematic_state_->getGlobalLinkTransform("xarmlink_eef")};
    Eigen::Matrix3d ppp {pos.linear()};
    Eigen::Matrix4d mat_2_ret;
    mat_2_ret << ppp(0,0), ppp(0,1), ppp(0,2), pos.translation().x(), 
        ppp(1,0), ppp(1,1), ppp(1,2), pos.translation().y(), 
        ppp(2,0), ppp(2,1), ppp(2,2), pos.translation().z(), 
        0.0, 0.0, 0.0, 1.0;
    return mat_2_ret;

    // mat_2_ret << ppp(0,0), ppp(0,1), ppp(0,2), pos.translation().x(), 
    //     ppp(1,0), ppp(1,1), ppp(1,2), pos.translation().y(), 
    //     ppp(2,0), ppp(2,1), ppp(2,2), pos.translation().z(), 
    //     0.0, 0.0, 0.0, 1.0;

    // Eigen::Quaternion<double> this_quat {this->ee_angular_velocities_.at(3),
    //                             this->ee_angular_velocities_.at(0),
    //                             this->ee_angular_velocities_.at(1),
    //                             this->ee_angular_velocities_.at(2)};

    // Eigen::Matrix<double, 3, 3> curr_rot = this_quat.toRotationMatrix();

    // Eigen::Matrix4d last_transform;
    // last_transform <<   curr_rot(0,0), curr_rot(0,1), curr_rot(0,2), this->ee_velocities_from_IMU_.at(0) * (1/50),
    //                     curr_rot(1,0), curr_rot(1,1), curr_rot(1,2), this->ee_velocities_from_IMU_.at(1) * (1/50),
    //                     curr_rot(2,0), curr_rot(2,1), curr_rot(2,2), this->ee_velocities_from_IMU_.at(2) * (1/50),
    //                     0, 0, 0, 1;
    
    // Eigen::Vector3d ahaha;
    // ahaha << 0.0, 1.5708, 0.0;
    // Eigen::AngleAxisd rollAngle(static_cast<double>(ahaha(0)), Eigen::Vector3d::UnitZ());
    // Eigen::AngleAxisd yawAngle(static_cast<double>(ahaha(1)), Eigen::Vector3d::UnitY());
    // Eigen::AngleAxisd pitchAngle(static_cast<double>(ahaha(2)), Eigen::Vector3d::UnitX());

    // Eigen::Quaterniond q {rollAngle * yawAngle * pitchAngle};
    // Eigen::Matrix3d rotation_matrix {q.matrix()};

    // Eigen::Matrix4d rot_imu_to_ee;
    // rot_imu_to_ee <<  rotation_matrix(0,0), rotation_matrix(0,1), rotation_matrix(0,2), 0,
    //                         rotation_matrix(1,0), rotation_matrix(1,1), rotation_matrix(1,2), 0,
    //                         rotation_matrix(2,0), rotation_matrix(2,1), rotation_matrix(2,2), 0,
    //                         0, 0, 0, 1;

    // Eigen::Matrix4d trans_imu_to_ee;
    // trans_imu_to_ee <<  1, 0, 0, 0.1,
    //                         0, 1, 0, 0,
    //                         0, 0, 1, 0.065,
    //                         0, 0, 0, 1;

    // Eigen::Matrix4d transform_imu_to_ee;
    // transform_imu_to_ee = (rot_imu_to_ee * trans_imu_to_ee);

    // Eigen::Matrix4d imu_in_ee_frame;
    // imu_in_ee_frame = last_transform * (transform_imu_to_ee.inverse());

    // return mat_2_ret * imu_in_ee_frame;
}

void ASDisturbanceController::update_base_pose_()
{
    Eigen::Quaternion<double> this_quat {this->current_base_IMU_measurement_.orientation.w,
                                this->current_base_IMU_measurement_.orientation.x,
                                this->current_base_IMU_measurement_.orientation.y,
                                this->current_base_IMU_measurement_.orientation.z};
    Eigen::Matrix<double, 3, 3> curr_rot = this_quat.toRotationMatrix();

    Eigen::Matrix4d disturbance_in_base_imu_frame; 
    disturbance_in_base_imu_frame <<    curr_rot(0,0), curr_rot(0,1), curr_rot(0,2), 0, 
                    curr_rot(1,0), curr_rot(1,1), curr_rot(1,2), this->y_velocity_ * (1 / 50), 
                    curr_rot(2,0), curr_rot(2,1), curr_rot(2,2), this->z_velocity_ * (1 / 50), 
                    0.0, 0.0, 0.0, 1.0;


    Eigen::Matrix4d base_2_imu_trans;
    base_2_imu_trans <<     1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0.2825,
                            0, 0, 0, 1; // need 2 fix z here ! 

    Eigen::Matrix4d buh;
    buh <<              1, 0, 0, -0.125,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1; // need 2 fix z here ! 

    Eigen::Matrix4d identity;
    identity <<     1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;

    Eigen::Matrix4d full_tf; 
    full_tf = identity * base_2_imu_trans * buh * disturbance_in_base_imu_frame;
    this->base_pose_ = ( full_tf.inverse() );
}

Eigen::Matrix4d ASDisturbanceController::get_ee_pose_wrt_base_footprint_frame()
{
    return this->base_pose_ * this->get_actual_ee_pose();
}

// convert homogeneous matrix into 6x1 vector of position and orientation
Eigen::VectorXd ASDisturbanceController::homog_to_trans_and_euler(Eigen::Matrix4d &mat)
{
    Eigen::Matrix3d rotation_element;
    rotation_element << mat(0,0), mat(0,1), mat(0,2), mat(1,0), mat(1,1), mat(1,2), mat(2,0), mat(2,1), mat(2,2);
    Eigen::Vector3d euler_angles {rotation_element.eulerAngles(0,1,2)};
    Eigen::VectorXd vec(6);
    vec << mat(0,3), mat(1,3), mat(2,3), euler_angles(0), euler_angles(1), euler_angles(2);
    return vec;
}

Eigen::VectorXd ASDisturbanceController::get_error_vector(Eigen::VectorXd &desired, Eigen::VectorXd &actual)
{
    // Correcting angle error:
    // https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
    Eigen::VectorXd err1 (6);
    err1 <<  (desired(0) - actual(0)), 
            (desired(1) - actual(1)), 
            (desired(2) - actual(2)), 
            ( this->custom_mod( ( ( desired(3) - actual(3) ) + M_PI_2 ), M_PI ) - M_PI_2 ), 
            ( this->custom_mod( ( ( desired(4) - actual(4) ) + M_PI_2 ), M_PI ) - M_PI_2 ), 
            ( this->custom_mod( ( ( desired(5) - actual(5) ) + M_PI_2 ), M_PI ) - M_PI_2 );

    // Eigen::VectorXd err1 (6);
    // err1 <<  (desired(0) - actual(0)), 
    //         (desired(1) - actual(1)), 
    //         (desired(2) - actual(2)), 
    //         0, 
    //         0, 
    //         0;


    this->publish_the_errors(err1);
    return err1;
}

void ASDisturbanceController::publish_the_errors(Eigen::VectorXd &err_six)
{
    geometry_msgs::PointStamped translation_error;
    translation_error.point.x = err_six(0);
    translation_error.point.y = err_six(1);
    translation_error.point.z = err_six(2);
    translation_error.header.stamp = ros::Time::now();
    this->trans_error_pub_.publish(translation_error);

    geometry_msgs::PointStamped orientation_error;
    orientation_error.point.x = err_six(3);
    orientation_error.point.y = err_six(4); 
    orientation_error.point.z = err_six(5);
    orientation_error.header.stamp = ros::Time::now();
    this->rot_error_pub_.publish(orientation_error);
}

Eigen::MatrixXd ASDisturbanceController::get_jacobian()
{
    Eigen::Vector3d reference_point_position(0.0, 0.0, 0.0);
    Eigen::MatrixXd jacobian;
    this->kinematic_state_->getJacobian(
        this->joint_model_group_,
        this->kinematic_state_->getLinkModel(
            this->joint_model_group_->getLinkModelNames().back()
        ),
        reference_point_position,
        jacobian
    );
    return jacobian;
}

Eigen::VectorXd ASDisturbanceController::get_delta_theta(Eigen::MatrixXd &this_jacobian, Eigen::VectorXd &current_error)
{
    Eigen::VectorXd d_theta (6);
    Eigen::MatrixXd buh = this_jacobian.completeOrthogonalDecomposition().pseudoInverse();
    d_theta = ( this->alpha_ * ( buh *  current_error ) );
    return d_theta;
}

void ASDisturbanceController::control_timestep()
{
    if(this->ctr_ > 1500)
    {
        Eigen::Matrix4d desired_pose_no_bumps {this->get_ee_pose_no_bumps()};
        Eigen::Matrix4d curr_pose {this->get_ee_pose_wrt_base_footprint_frame()};
        Eigen::VectorXd curr_error (6);
        Eigen::VectorXd pos_no_bumps {this->homog_to_trans_and_euler(desired_pose_no_bumps)};
        Eigen::VectorXd curr {this->homog_to_trans_and_euler(curr_pose)};
        curr_error = this->get_error_vector(pos_no_bumps, curr);
        Eigen::MatrixXd j;
        j = this->get_jacobian();
        Eigen::VectorXd delta_theta;
        delta_theta = this->get_delta_theta(j, curr_error);
        this->correct_joint_positions(delta_theta); // this is the control line ! 
    }
}

void ASDisturbanceController::correct_joint_positions(Eigen::VectorXd &dee_theta)
{
    double factor {0.05}; // this works with base IMU
    for(int o {0}; o < 6; o++)
    {
        this->corrected_joint_positions_.at(o) = this->corrected_joint_positions_.at(o) + dee_theta(o);
        if (this->corrected_joint_positions_.at(o) < 0)
        {
            if (this->corrected_joint_positions_.at(o) < this->desired_joint_positions_.at(o) - (M_PI * factor))
            {
                this->corrected_joint_positions_.at(o) = this->desired_joint_positions_.at(o) - (M_PI * factor);
            } else if (this->corrected_joint_positions_.at(o) > this->desired_joint_positions_.at(o) + (M_PI * factor))
            {
                this->corrected_joint_positions_.at(o) = this->desired_joint_positions_.at(o) + (M_PI * factor);
            }
        } else if (this->corrected_joint_positions_.at(o) > 0)
        {
            if (this->corrected_joint_positions_.at(o) > this->desired_joint_positions_.at(o) + (M_PI * factor))
            {
                this->corrected_joint_positions_.at(o) = this->desired_joint_positions_.at(o) + (M_PI * factor);
            } else if (this->corrected_joint_positions_.at(o) < this->desired_joint_positions_.at(o) - (M_PI * factor))
            {
                this->corrected_joint_positions_.at(o) = this->desired_joint_positions_.at(o) - (M_PI * factor);
            }
        }
    }
}


















