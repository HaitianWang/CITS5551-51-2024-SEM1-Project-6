<template>
  <div>
    <el-form ref="regiForm" :model="user" label-width="120px" class="login-box">
      <h3 class="regi-title">欢迎注册</h3>
      <el-form-item label="账号" prop="username">
        <el-input type="text" placeholder="请输入账号" v-model="user.username"/>
      </el-form-item>
      <el-form-item label="密码" prop="pwd">
        <el-input type="password" placeholder="请输入密码（6位以上）" v-model="user.pwd"/>
      </el-form-item>
      <el-form-item label="重复密码" prop="pwd2">
        <el-input type="password" placeholder="请重复密码" v-model="user.pwd2"/>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" v-on:click="regi('regiForm')" id="btn">确认</el-button>
        <el-button type="primary" v-on:click="goFrontPage()" id="btn2">返回</el-button>
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
            window.alert("请输入用户名！")
          }
          else if(this.user.pwd.trim().length==0){
            window.alert("请输入密码！")
          }
          else if(this.user.pwd.length>0 && this.user.pwd.length<6){
            window.alert("密码需要六位以上！")
          }
          else if(this.user.pwd!=this.user.pwd2){
            window.alert("重复密码不一致！")
          }
          else {
            let param = new FormData(); //创建form对象
            param.append("username", this.user.username)
            param.append("pwd", this.user.pwd)
            let config = {
              headers: {"Content-Type": "multipart/form-data"},
            }; //添加请求头
            axios
              .post(this.server_url + "/regi", param, config)
              .then((response) => {
                console.log(response.data)
                if (response.data == '1') {
                  window.alert("注册成功！")
                  this.$router.replace({
                    path: "/frontPage",
                    name: 'frontPage',
                    query: {
                      username: this.user.username
                    }
                  })
                } else {
                  if(response.data == '0'){
                    window.alert("该用户名已经存在！")
                    this.$refs['regiForm'].resetFields();
                  }
                  else {
                    window.alert("注册失败")
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
