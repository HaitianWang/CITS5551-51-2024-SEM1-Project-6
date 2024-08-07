<template>
  <div class="login-container">
    <el-form ref="loginForm" label-width="80px" class="login-box">
      <h3 class="login-title">UWAIntelliCrop</h3>
      <el-form-item label="username" prop="username">
        <el-input type="text" placeholder="please enter username" v-model="username"/>
      </el-form-item>
      <el-form-item label="password" prop="pwd">
        <el-input type="password" placeholder="please enter password" v-model="pwd"/>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" v-on:click="login('loginForm')" id="btn">login</el-button>
      </el-form-item>
      <h5>no account? please <span v-on:click="goRegister()" style="color: blue;text-decoration: underline">register</span></h5>
    </el-form>
  </div>
</template>

<script>
  import axios from "axios";

  export default {
    name: "frontPage",
    created() {
      this.$emit('header', false);
      this.$emit('content', false);
      this.$emit('footer', false);
    },
    data() {
      return {
        username: "",
        pwd: "",
        server_url: "http://127.0.0.1:5003"
      }
    },
    methods: {
      login(data){
        let param = new FormData(); // 创建 form 对象
        param.append("username", this.username);
        param.append("pwd", this.pwd);
        let config = {
          headers: { "Content-Type": "multipart/form-data" },
        }; // 添加请求头
        axios
          .post(this.server_url + "/login", param, config)
          .then((response) => {
            console.log(response.data)
            if(response.data=='1'){
              this.$router.replace({
                path: "/mainPage",
                name: 'mainPage',
                query: {
                  username:this.username
                }
              });
            }
            else {
              window.alert("username or password error")
            }
          });
      },
      goRegister(){
        this.$router.push({
          path: "/regiPage",
          name: "regiPage",
        });
      }
    },
  };
</script>

<style scoped>
  .login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background-image: url('~@/assets/6.jpg'); /* 使用 ~@ 以确保路径正确解析 */
    background-size: cover;
    background-position: center;
  }

  .login-box {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #DCDFE6;
    width: 350px;
    padding: 35px 35px 15px 35px;
    border-radius: 5px;
    box-shadow: 0 0 25px #909399;
  }

  #btn {
    margin-left: -60px;
  }
</style>
