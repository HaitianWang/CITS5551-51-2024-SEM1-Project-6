<template>
  <div>
    <el-form ref="regiForm" :model="user" label-width="120px" class="login-box">
      <h3 class="regi-title">Welcome to Register</h3>
      <el-form-item label="Username" prop="username">
        <el-input type="text" placeholder="Please enter username" v-model="user.username"/>
      </el-form-item>
      <el-form-item label="Password" prop="pwd">
        <el-input type="password" placeholder="Please enter password (more than 6 characters)" v-model="user.pwd"/>
      </el-form-item>
      <el-form-item label="Repeat Password" prop="pwd2">
        <el-input type="password" placeholder="Please repeat the password" v-model="user.pwd2"/>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" v-on:click="regi('regiForm')" id="btn">Confirm</el-button>
        <el-button type="primary" v-on:click="goFrontPage()" id="btn2">Back</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script>
  import axios from "axios";

  export default {
    name: "regiPage",
    created() {
      this.$emit('header', false);
      this.$emit('content', false);
      this.$emit('footer', false);
    },
    data() {
      return {
        user: {
          username: "",
          pwd: "",
          pwd2: ""
        },
        server_url: "http://127.0.0.1:5003"
      }
    },
    methods: {
      regi(data) {
        if(this.user.username.trim().length==0){
          window.alert("Please enter username!")
        }
        else if(this.user.pwd.trim().length==0){
          window.alert("Please enter password!")
        }
        else if(this.user.pwd.length>0 && this.user.pwd.length<6){
          window.alert("Password must be more than 6 characters!")
        }
        else if(this.user.pwd!=this.user.pwd2){
          window.alert("Passwords do not match!")
        }
        else {
          let param = new FormData(); // create form object
          param.append("username", this.user.username)
          param.append("pwd", this.user.pwd)
          let config = {
            headers: {"Content-Type": "multipart/form-data"},
          }; // add request headers
          axios
            .post(this.server_url + "/regi", param, config)
            .then((response) => {
              console.log(response.data)
              if (response.data == '1') {
                window.alert("Registration successful!")
                this.$router.replace({
                  path: "/frontPage",
                  name: 'frontPage',
                  query: {
                    username: this.user.username
                  }
                })
              } else {
                if(response.data == '0'){
                  window.alert("Username already exists!")
                  this.$refs['regiForm'].resetFields();
                }
                else {
                  window.alert("Registration failed")
                }
              }
            });
        }
      },
      goFrontPage(){
        this.$router.push({
          path: "/frontPage",
          name: "frontPage",
        })
      }
    }
  }
</script>

<style scoped>
  .login-box {
    border: 1px solid #DCDFE6;
    width: 600px;
    margin: 180px auto;
    padding: 35px 35px 15px 35px;
    border-radius: 5px;
    -webkit-border-radius: 5px;
    -moz-border-radius: 5px;
    box-shadow: 0 0 25px #909399;
  }
  #btn{
    margin-left: -100px;
  }
</style>
